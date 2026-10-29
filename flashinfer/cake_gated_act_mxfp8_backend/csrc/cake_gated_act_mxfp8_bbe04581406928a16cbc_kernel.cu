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
#define SMEM_TILE_STAGE_BYTES 32768
#define SMEM_TILE_STRIDE 32768
#define SMEM_TOTAL 32768
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_gated_act_mxfp8_bbe04581406928a16cbc(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 128 + row;
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
    int grow_b = grow + 32;
    int grow_c = grow + 64;
    int grow_d = grow + 96;
    int gate_index_3 = grow_b * (2 * K) + col;
    unsigned int vec_load[8];
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_3 + (0))) : "memory");
        vec_load[0 + 0] = (uint32_t)_iv_2_0;
        vec_load[0 + 1] = (uint32_t)_iv_2_1;
        vec_load[0 + 2] = (uint32_t)_iv_2_2;
        vec_load[0 + 3] = (uint32_t)_iv_2_3;
        vec_load[0 + 4] = (uint32_t)_iv_2_4;
        vec_load[0 + 5] = (uint32_t)_iv_2_5;
        vec_load[0 + 6] = (uint32_t)_iv_2_6;
        vec_load[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int vec_load_1[8];
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_3 + K) + (0))) : "memory");
        vec_load_1[0 + 0] = (uint32_t)_iv_3_0;
        vec_load_1[0 + 1] = (uint32_t)_iv_3_1;
        vec_load_1[0 + 2] = (uint32_t)_iv_3_2;
        vec_load_1[0 + 3] = (uint32_t)_iv_3_3;
        vec_load_1[0 + 4] = (uint32_t)_iv_3_4;
        vec_load_1[0 + 5] = (uint32_t)_iv_3_5;
        vec_load_1[0 + 6] = (uint32_t)_iv_3_6;
        vec_load_1[0 + 7] = (uint32_t)_iv_3_7;
    }
    int gate_index_4 = grow_c * (2 * K) + col;
    unsigned int vec_load_0[8];
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + gate_index_4 + (0))) : "memory");
        vec_load_0[0 + 0] = (uint32_t)_iv_4_0;
        vec_load_0[0 + 1] = (uint32_t)_iv_4_1;
        vec_load_0[0 + 2] = (uint32_t)_iv_4_2;
        vec_load_0[0 + 3] = (uint32_t)_iv_4_3;
        vec_load_0[0 + 4] = (uint32_t)_iv_4_4;
        vec_load_0[0 + 5] = (uint32_t)_iv_4_5;
        vec_load_0[0 + 6] = (uint32_t)_iv_4_6;
        vec_load_0[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int vec_load_1_0[8];
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
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(gated_input + (gate_index_4 + K) + (0))) : "memory");
        vec_load_1_0[0 + 0] = (uint32_t)_iv_5_0;
        vec_load_1_0[0 + 1] = (uint32_t)_iv_5_1;
        vec_load_1_0[0 + 2] = (uint32_t)_iv_5_2;
        vec_load_1_0[0 + 3] = (uint32_t)_iv_5_3;
        vec_load_1_0[0 + 4] = (uint32_t)_iv_5_4;
        vec_load_1_0[0 + 5] = (uint32_t)_iv_5_5;
        vec_load_1_0[0 + 6] = (uint32_t)_iv_5_6;
        vec_load_1_0[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_5 = grow_d * (2 * K) + col;
    unsigned int vec_load_2[8];
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
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_5 + (0))) : "memory");
        vec_load_2[0 + 0] = (uint32_t)_iv_6_0;
        vec_load_2[0 + 1] = (uint32_t)_iv_6_1;
        vec_load_2[0 + 2] = (uint32_t)_iv_6_2;
        vec_load_2[0 + 3] = (uint32_t)_iv_6_3;
        vec_load_2[0 + 4] = (uint32_t)_iv_6_4;
        vec_load_2[0 + 5] = (uint32_t)_iv_6_5;
        vec_load_2[0 + 6] = (uint32_t)_iv_6_6;
        vec_load_2[0 + 7] = (uint32_t)_iv_6_7;
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
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_5 + K) + (0))) : "memory");
        vec_load_1_1[0 + 0] = (uint32_t)_iv_7_0;
        vec_load_1_1[0 + 1] = (uint32_t)_iv_7_1;
        vec_load_1_1[0 + 2] = (uint32_t)_iv_7_2;
        vec_load_1_1[0 + 3] = (uint32_t)_iv_7_3;
        vec_load_1_1[0 + 4] = (uint32_t)_iv_7_4;
        vec_load_1_1[0 + 5] = (uint32_t)_iv_7_5;
        vec_load_1_1[0 + 6] = (uint32_t)_iv_7_6;
        vec_load_1_1[0 + 7] = (uint32_t)_iv_7_7;
    }
    unsigned int packed[8];
    unsigned int packed_b[8];
    unsigned int packed_c[8];
    unsigned int packed_d[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[0]));
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
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_9;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_8 = __all_sync(0xFFFFFFFF, _min_9 > -87.0f);
    if (_warp_vote_8 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0), "f"(_fma_11), "f"(r0));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r1), "f"(_fma_13), "f"(r1));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(cvt_f32, make_float2(_fma_10, _fma_12));
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_0 = approx_exp2((-cvt_f32_0.x) * 1.442695f);
        float e1_0 = approx_exp2((-cvt_f32_0.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r0_0), "f"(_fma_16), "f"(r0_0));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_0), "f"(_fma_18), "f"(r1_0));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_0, make_float2(_fma_15, _fma_17));
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_1 = approx_exp2((-cvt_f32_2.x) * 1.442695f);
        float e1_1 = approx_exp2((-cvt_f32_2.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_1), "f"(_fma_21), "f"(r0_1));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r1_1), "f"(_fma_23), "f"(r1_1));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(cvt_f32_2, make_float2(_fma_20, _fma_22));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_2 = approx_exp2((-cvt_f32_3.x) * 1.442695f);
        float e1_2 = approx_exp2((-cvt_f32_3.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r0_2), "f"(_fma_26), "f"(r0_2));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_2), "f"(_fma_28), "f"(r1_2));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_3, make_float2(_fma_25, _fma_27));
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_3 = approx_exp2((-cvt_f32_4.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_4.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_3), "f"(_fma_31), "f"(r0_3));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r1_3), "f"(_fma_33), "f"(r1_3));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(cvt_f32_4, make_float2(_fma_30, _fma_32));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_4 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r0_4), "f"(_fma_36), "f"(r0_4));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r1_4), "f"(_fma_38), "f"(r1_4));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(cvt_f32_5, make_float2(_fma_35, _fma_37));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_5 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r0_5), "f"(_fma_41), "f"(r0_5));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r1_5), "f"(_fma_43), "f"(r1_5));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(cvt_f32_6, make_float2(_fma_40, _fma_42));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_6 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_6 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(r0_6), "f"(_fma_46), "f"(r0_6));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r1_6), "f"(_fma_48), "f"(r1_6));
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(cvt_f32_7, make_float2(_fma_45, _fma_47));
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float e0_7 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(r0_7), "f"(_fma_51), "f"(r0_7));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(r1_7), "f"(_fma_53), "f"(r1_7));
        float2 _f32x2_mul_a_54 = mul_f32x2_noftz(cvt_f32_8, make_float2(_fma_50, _fma_52));
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float e0_8 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(r0_8), "f"(_fma_56), "f"(r0_8));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(r1_8), "f"(_fma_58), "f"(r1_8));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(cvt_f32_9, make_float2(_fma_55, _fma_57));
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float e0_9 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(r0_9), "f"(_fma_61), "f"(r0_9));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(r1_9), "f"(_fma_63), "f"(r1_9));
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(cvt_f32_10, make_float2(_fma_60, _fma_62));
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float e0_10 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(r0_10), "f"(_fma_66), "f"(r0_10));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(r1_10), "f"(_fma_68), "f"(r1_10));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(cvt_f32_11, make_float2(_fma_65, _fma_67));
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float e0_11 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(r0_11), "f"(_fma_71), "f"(r0_11));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(r1_11), "f"(_fma_73), "f"(r1_11));
        float2 _f32x2_mul_a_74 = mul_f32x2_noftz(cvt_f32_12, make_float2(_fma_70, _fma_72));
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float e0_12 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(r0_12), "f"(_fma_76), "f"(r0_12));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(r1_12), "f"(_fma_78), "f"(r1_12));
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(cvt_f32_13, make_float2(_fma_75, _fma_77));
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float e0_13 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(r0_13), "f"(_fma_81), "f"(r0_13));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(r1_13), "f"(_fma_83), "f"(r1_13));
        float2 _f32x2_mul_a_84 = mul_f32x2_noftz(cvt_f32_14, make_float2(_fma_80, _fma_82));
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float e0_14 = approx_exp2((-cvt_f32_15.x) * 1.442695f);
        float e1_14 = approx_exp2((-cvt_f32_15.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(r0_14), "f"(_fma_86), "f"(r0_14));
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(r1_14), "f"(_fma_88), "f"(r1_14));
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(cvt_f32_15, make_float2(_fma_85, _fma_87));
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float e0_15 = approx_exp2((-cvt_f32_16.x) * 1.442695f);
        float e1_15 = approx_exp2((-cvt_f32_16.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(r0_15), "f"(_fma_91), "f"(r0_15));
        float _fma_93;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(r1_15), "f"(_fma_93), "f"(r1_15));
        float2 _f32x2_mul_a_94 = mul_f32x2_noftz(cvt_f32_16, make_float2(_fma_90, _fma_92));
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_15.x, mul_15.y)));
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float e0_16 = approx_exp2((-cvt_f32_17.x) * 1.442695f);
        float e1_16 = approx_exp2((-cvt_f32_17.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(r0_16), "f"(_fma_96), "f"(r0_16));
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(r1_16), "f"(_fma_98), "f"(r1_16));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(cvt_f32_17, make_float2(_fma_95, _fma_97));
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_16.x, mul_16.y)));
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float e0_17 = approx_exp2((-cvt_f32_18.x) * 1.442695f);
        float e1_17 = approx_exp2((-cvt_f32_18.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(r0_17), "f"(_fma_101), "f"(r0_17));
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(r1_17), "f"(_fma_103), "f"(r1_17));
        float2 _f32x2_mul_a_104 = mul_f32x2_noftz(cvt_f32_18, make_float2(_fma_100, _fma_102));
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_17.x, mul_17.y)));
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float e0_18 = approx_exp2((-cvt_f32_19.x) * 1.442695f);
        float e1_18 = approx_exp2((-cvt_f32_19.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(r0_18), "f"(_fma_106), "f"(r0_18));
        float _fma_108;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(r1_18), "f"(_fma_108), "f"(r1_18));
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(cvt_f32_19, make_float2(_fma_105, _fma_107));
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_18.x, mul_18.y)));
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float e0_19 = approx_exp2((-cvt_f32_20.x) * 1.442695f);
        float e1_19 = approx_exp2((-cvt_f32_20.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(r0_19), "f"(_fma_111), "f"(r0_19));
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(r1_19), "f"(_fma_113), "f"(r1_19));
        float2 _f32x2_mul_a_114 = mul_f32x2_noftz(cvt_f32_20, make_float2(_fma_110, _fma_112));
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_19.x, mul_19.y)));
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float e0_20 = approx_exp2((-cvt_f32_21.x) * 1.442695f);
        float e1_20 = approx_exp2((-cvt_f32_21.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(r0_20), "f"(_fma_116), "f"(r0_20));
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(r1_20), "f"(_fma_118), "f"(r1_20));
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(cvt_f32_21, make_float2(_fma_115, _fma_117));
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_20.x, mul_20.y)));
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float e0_21 = approx_exp2((-cvt_f32_22.x) * 1.442695f);
        float e1_21 = approx_exp2((-cvt_f32_22.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(r0_21), "f"(_fma_121), "f"(r0_21));
        float _fma_123;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(r1_21), "f"(_fma_123), "f"(r1_21));
        float2 _f32x2_mul_a_124 = mul_f32x2_noftz(cvt_f32_22, make_float2(_fma_120, _fma_122));
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_21.x, mul_21.y)));
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float e0_22 = approx_exp2((-cvt_f32_23.x) * 1.442695f);
        float e1_22 = approx_exp2((-cvt_f32_23.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(r0_22), "f"(_fma_126), "f"(r0_22));
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(r1_22), "f"(_fma_128), "f"(r1_22));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(cvt_f32_23, make_float2(_fma_125, _fma_127));
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_22.x, mul_22.y)));
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float e0_23 = approx_exp2((-cvt_f32_24.x) * 1.442695f);
        float e1_23 = approx_exp2((-cvt_f32_24.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(r0_23), "f"(_fma_131), "f"(r0_23));
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(r1_23), "f"(_fma_133), "f"(r1_23));
        float2 _f32x2_mul_a_134 = mul_f32x2_noftz(cvt_f32_24, make_float2(_fma_130, _fma_132));
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_23.x, mul_23.y)));
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float e0_24 = approx_exp2((-cvt_f32_25.x) * 1.442695f);
        float e1_24 = approx_exp2((-cvt_f32_25.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(r0_24), "f"(_fma_136), "f"(r0_24));
        float _fma_138;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(r1_24), "f"(_fma_138), "f"(r1_24));
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(cvt_f32_25, make_float2(_fma_135, _fma_137));
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_24.x, mul_24.y)));
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float e0_25 = approx_exp2((-cvt_f32_26.x) * 1.442695f);
        float e1_25 = approx_exp2((-cvt_f32_26.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(r0_25), "f"(_fma_141), "f"(r0_25));
        float _fma_143;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(r1_25), "f"(_fma_143), "f"(r1_25));
        float2 _f32x2_mul_a_144 = mul_f32x2_noftz(cvt_f32_26, make_float2(_fma_140, _fma_142));
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_25.x, mul_25.y)));
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float e0_26 = approx_exp2((-cvt_f32_27.x) * 1.442695f);
        float e1_26 = approx_exp2((-cvt_f32_27.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(r0_26), "f"(_fma_146), "f"(r0_26));
        float _fma_148;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(r1_26), "f"(_fma_148), "f"(r1_26));
        float2 _f32x2_mul_a_149 = mul_f32x2_noftz(cvt_f32_27, make_float2(_fma_145, _fma_147));
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_26.x, mul_26.y)));
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float e0_27 = approx_exp2((-cvt_f32_28.x) * 1.442695f);
        float e1_27 = approx_exp2((-cvt_f32_28.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(r0_27), "f"(_fma_151), "f"(r0_27));
        float _fma_153;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(r1_27), "f"(_fma_153), "f"(r1_27));
        float2 _f32x2_mul_a_154 = mul_f32x2_noftz(cvt_f32_28, make_float2(_fma_150, _fma_152));
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_27.x, mul_27.y)));
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float e0_28 = approx_exp2((-cvt_f32_29.x) * 1.442695f);
        float e1_28 = approx_exp2((-cvt_f32_29.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(r0_28), "f"(_fma_156), "f"(r0_28));
        float _fma_158;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(r1_28), "f"(_fma_158), "f"(r1_28));
        float2 _f32x2_mul_a_159 = mul_f32x2_noftz(cvt_f32_29, make_float2(_fma_155, _fma_157));
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_28.x, mul_28.y)));
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float e0_29 = approx_exp2((-cvt_f32_30.x) * 1.442695f);
        float e1_29 = approx_exp2((-cvt_f32_30.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(r0_29), "f"(_fma_161), "f"(r0_29));
        float _fma_163;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(r1_29), "f"(_fma_163), "f"(r1_29));
        float2 _f32x2_mul_a_164 = mul_f32x2_noftz(cvt_f32_30, make_float2(_fma_160, _fma_162));
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_29.x, mul_29.y)));
        float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float e0_30 = approx_exp2((-cvt_f32_31.x) * 1.442695f);
        float e1_30 = approx_exp2((-cvt_f32_31.y) * 1.442695f);
        float d0_30 = 1.0f + e0_30;
        float d1_30 = 1.0f + e1_30;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(r0_30), "f"(_fma_166), "f"(r0_30));
        float _fma_168;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(r1_30), "f"(_fma_168), "f"(r1_30));
        float2 _f32x2_mul_a_169 = mul_f32x2_noftz(cvt_f32_31, make_float2(_fma_165, _fma_167));
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&cvt_f32_1_30));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_30.x, mul_30.y)));
    } else {
        int band = 0;
        float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_31 = approx_exp2((-cvt_f32_32.x) * 1.442695f);
        float e1_31 = approx_exp2((-cvt_f32_32.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_31), "f"(_fma_170), "f"(r0_31));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_31), "f"(_fma_171), "f"(r1_31));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_172 = mul_f32x2_noftz(cvt_f32_32, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&cvt_f32_1_31));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_31.x, mul_31.y)));
        float2 cvt_f32_32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_31_1 = approx_exp2((-cvt_f32_32_1.x) * 1.442695f);
        float e1_31_1 = approx_exp2((-cvt_f32_32_1.y) * 1.442695f);
        float d0_31_1 = 1.0f + e0_31_1;
        float d1_31_1 = 1.0f + e1_31_1;
        float r0_31_1 = approx_rcp(d0_31_1);
        float r1_31_1 = approx_rcp(d1_31_1);
        float _fma_173;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(-d0_31_1), "f"(r0_31_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_31_1), "f"(_fma_173), "f"(r0_31_1));
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d1_31_1), "f"(r1_31_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_31_1), "f"(_fma_174), "f"(r1_31_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_175 = mul_f32x2_noftz(cvt_f32_32_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&cvt_f32_1_31_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_31_1.x, mul_31_1.y)));
        float2 cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_32 = approx_exp2((-cvt_f32_33.x) * 1.442695f);
        float e1_32 = approx_exp2((-cvt_f32_33.y) * 1.442695f);
        float d0_32 = 1.0f + e0_32;
        float d1_32 = 1.0f + e1_32;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float _fma_176;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(-d0_32), "f"(r0_32), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_32), "f"(_fma_176), "f"(r0_32));
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d1_32), "f"(r1_32), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_32), "f"(_fma_177), "f"(r1_32));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_178 = mul_f32x2_noftz(cvt_f32_33, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&cvt_f32_1_32));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_32.x, mul_32.y)));
        float2 cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_33 = approx_exp2((-cvt_f32_34.x) * 1.442695f);
        float e1_33 = approx_exp2((-cvt_f32_34.y) * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_179;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_33), "f"(_fma_179), "f"(r0_33));
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_33), "f"(_fma_180), "f"(r1_33));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_181 = mul_f32x2_noftz(cvt_f32_34, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&cvt_f32_1_33));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_33.x, mul_33.y)));
        float2 cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_34 = approx_exp2((-cvt_f32_35.x) * 1.442695f);
        float e1_34 = approx_exp2((-cvt_f32_35.y) * 1.442695f);
        float d0_34 = 1.0f + e0_34;
        float d1_34 = 1.0f + e1_34;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float _fma_182;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(-d0_34), "f"(r0_34), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_34), "f"(_fma_182), "f"(r0_34));
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(-d1_34), "f"(r1_34), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_34), "f"(_fma_183), "f"(r1_34));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_184 = mul_f32x2_noftz(cvt_f32_35, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&cvt_f32_1_34));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_34.x, mul_34.y)));
        float2 cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_35 = approx_exp2((-cvt_f32_36.x) * 1.442695f);
        float e1_35 = approx_exp2((-cvt_f32_36.y) * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_185;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_35), "f"(_fma_185), "f"(r0_35));
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_35), "f"(_fma_186), "f"(r1_35));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_187 = mul_f32x2_noftz(cvt_f32_36, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&cvt_f32_1_35));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_35.x, mul_35.y)));
        float2 cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_36 = approx_exp2((-cvt_f32_37.x) * 1.442695f);
        float e1_36 = approx_exp2((-cvt_f32_37.y) * 1.442695f);
        float d0_36 = 1.0f + e0_36;
        float d1_36 = 1.0f + e1_36;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float _fma_188;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(-d0_36), "f"(r0_36), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_36), "f"(_fma_188), "f"(r0_36));
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d1_36), "f"(r1_36), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_36), "f"(_fma_189), "f"(r1_36));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_190 = mul_f32x2_noftz(cvt_f32_37, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&cvt_f32_1_36));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_36.x, mul_36.y)));
        float2 cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_37 = approx_exp2((-cvt_f32_38.x) * 1.442695f);
        float e1_37 = approx_exp2((-cvt_f32_38.y) * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_191;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_37), "f"(_fma_191), "f"(r0_37));
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_37), "f"(_fma_192), "f"(r1_37));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_193 = mul_f32x2_noftz(cvt_f32_38, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&cvt_f32_1_37));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_37.x, mul_37.y)));
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_8[1];
                    {
                        _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_9[1];
                    {
                        _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    float2 cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                    float2 cvt_f32_1_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                    float e0_38 = approx_exp2((-cvt_f32_39.x) * 1.442695f);
                    float e1_38 = approx_exp2((-cvt_f32_39.y) * 1.442695f);
                    float d0_38 = 1.0f + e0_38;
                    float d1_38 = 1.0f + e1_38;
                    float r0_38 = approx_rcp(d0_38);
                    float r1_38 = approx_rcp(d1_38);
                    float _fma_194;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(-d0_38), "f"(r0_38), "f"(1.0f));
                    float fma_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_38), "f"(_fma_194), "f"(r0_38));
                    float _fma_195;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d1_38), "f"(r1_38), "f"(1.0f));
                    float fma_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_38), "f"(_fma_195), "f"(r1_38));
                    float t0_7 = fma_8;
                    float t1_7 = fma_1_7;
                    float d4_8a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_38), "f"(0.25f));
                    float r4_8a = approx_rcp(d4_8a);
                    float _fma_197;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
                    float _fma_196;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(r4_8a), "f"(_fma_197), "f"(r4_8a));
                    float big_8a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_196, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(big_8a), "f"(8.388608e+06f));
                    float whole_8a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_198), "f"(8.388608e+06f));
                    float above_8a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_200;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_200) : "f"(big_8a), "f"(whole_8a));
                    float _abs_199;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_199) : "f"(_sub_200));
                    float _add_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(big_8a), "f"(0.5f));
                    float _sub_202;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_202) : "f"(big_8a), "f"(0.5f));
                    float rounded_8a = ((_abs_199 == 0.5f) ? ((above_8a > 0.0f) ? _add_201 : _sub_202) : whole_8a);
                    float u0 = ((d0_38 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0_7);
                    float d4_8a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_38), "f"(0.25f));
                    float r4_8a_0 = approx_rcp(d4_8a_0);
                    float _fma_204;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
                    float _fma_203;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(r4_8a_0), "f"(_fma_204), "f"(r4_8a_0));
                    float big_8a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_203, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(big_8a_0), "f"(8.388608e+06f));
                    float whole_8a_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_205), "f"(8.388608e+06f));
                    float above_8a_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_207;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_207) : "f"(big_8a_0), "f"(whole_8a_0));
                    float _abs_206;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_206) : "f"(_sub_207));
                    float _add_208;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(big_8a_0), "f"(0.5f));
                    float _sub_209;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_209) : "f"(big_8a_0), "f"(0.5f));
                    float rounded_8a_0 = ((_abs_206 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_208 : _sub_209) : whole_8a_0);
                    float u1 = ((d1_38 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1_7);
                    float2 _f32x2_mul_a_210 = mul_f32x2_noftz(cvt_f32_39, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
                    float2 mul_38;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&cvt_f32_1_38));
                    unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(mul_38.x, mul_38.y)));
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
        int band_0 = 0;
        float2 cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float e0_39 = approx_exp2((-cvt_f32_40.x) * 1.442695f);
        float e1_39 = approx_exp2((-cvt_f32_40.y) * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_211;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_39), "f"(_fma_211), "f"(r0_39));
        float _fma_212;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_39), "f"(_fma_212), "f"(r1_39));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        band_0 = band_0 | ((fminf(t0_8, t1_8) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_213 = mul_f32x2_noftz(cvt_f32_40, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&cvt_f32_1_39));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_39.x, mul_39.y)));
        float2 cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float e0_40 = approx_exp2((-cvt_f32_41.x) * 1.442695f);
        float e1_40 = approx_exp2((-cvt_f32_41.y) * 1.442695f);
        float d0_40 = 1.0f + e0_40;
        float d1_40 = 1.0f + e1_40;
        float r0_40 = approx_rcp(d0_40);
        float r1_40 = approx_rcp(d1_40);
        float _fma_214;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(-d0_40), "f"(r0_40), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_40), "f"(_fma_214), "f"(r0_40));
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d1_40), "f"(r1_40), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_40), "f"(_fma_215), "f"(r1_40));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        band_0 = band_0 | ((fminf(t0_9, t1_9) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_216 = mul_f32x2_noftz(cvt_f32_41, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&cvt_f32_1_40));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_40.x, mul_40.y)));
        float2 cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float e0_41 = approx_exp2((-cvt_f32_42.x) * 1.442695f);
        float e1_41 = approx_exp2((-cvt_f32_42.y) * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_217;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_41), "f"(_fma_217), "f"(r0_41));
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_41), "f"(_fma_218), "f"(r1_41));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        band_0 = band_0 | ((fminf(t0_10, t1_10) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_219 = mul_f32x2_noftz(cvt_f32_42, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&cvt_f32_1_41));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_41.x, mul_41.y)));
        float2 cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float e0_42 = approx_exp2((-cvt_f32_43.x) * 1.442695f);
        float e1_42 = approx_exp2((-cvt_f32_43.y) * 1.442695f);
        float d0_42 = 1.0f + e0_42;
        float d1_42 = 1.0f + e1_42;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float _fma_220;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(-d0_42), "f"(r0_42), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_42), "f"(_fma_220), "f"(r0_42));
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(-d1_42), "f"(r1_42), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_42), "f"(_fma_221), "f"(r1_42));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        band_0 = band_0 | ((fminf(t0_11, t1_11) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_222 = mul_f32x2_noftz(cvt_f32_43, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&cvt_f32_1_42));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_42.x, mul_42.y)));
        float2 cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float e0_43 = approx_exp2((-cvt_f32_44.x) * 1.442695f);
        float e1_43 = approx_exp2((-cvt_f32_44.y) * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_223;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_43), "f"(_fma_223), "f"(r0_43));
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_43), "f"(_fma_224), "f"(r1_43));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        band_0 = band_0 | ((fminf(t0_12, t1_12) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_225 = mul_f32x2_noftz(cvt_f32_44, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&cvt_f32_1_43));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_43.x, mul_43.y)));
        float2 cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float e0_44 = approx_exp2((-cvt_f32_45.x) * 1.442695f);
        float e1_44 = approx_exp2((-cvt_f32_45.y) * 1.442695f);
        float d0_44 = 1.0f + e0_44;
        float d1_44 = 1.0f + e1_44;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float _fma_226;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(-d0_44), "f"(r0_44), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_44), "f"(_fma_226), "f"(r0_44));
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(-d1_44), "f"(r1_44), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_44), "f"(_fma_227), "f"(r1_44));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        band_0 = band_0 | ((fminf(t0_13, t1_13) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_228 = mul_f32x2_noftz(cvt_f32_45, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&cvt_f32_1_44));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_44.x, mul_44.y)));
        float2 cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float e0_45 = approx_exp2((-cvt_f32_46.x) * 1.442695f);
        float e1_45 = approx_exp2((-cvt_f32_46.y) * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_229;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_45), "f"(_fma_229), "f"(r0_45));
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_45), "f"(_fma_230), "f"(r1_45));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        band_0 = band_0 | ((fminf(t0_14, t1_14) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_231 = mul_f32x2_noftz(cvt_f32_46, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&cvt_f32_1_45));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_45.x, mul_45.y)));
        float2 cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float e0_46 = approx_exp2((-cvt_f32_47.x) * 1.442695f);
        float e1_46 = approx_exp2((-cvt_f32_47.y) * 1.442695f);
        float d0_46 = 1.0f + e0_46;
        float d1_46 = 1.0f + e1_46;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float _fma_232;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(-d0_46), "f"(r0_46), "f"(1.0f));
        float fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_16) : "f"(r0_46), "f"(_fma_232), "f"(r0_46));
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d1_46), "f"(r1_46), "f"(1.0f));
        float fma_1_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_15) : "f"(r1_46), "f"(_fma_233), "f"(r1_46));
        float t0_15 = fma_16;
        float t1_15 = fma_1_15;
        band_0 = band_0 | ((fminf(t0_15, t1_15) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_234 = mul_f32x2_noftz(cvt_f32_47, make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f)));
        float2 mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&cvt_f32_1_46));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_46.x, mul_46.y)));
        if (band_0 != 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 8; i_1++) {
                if ((band_0 >> i_1 & 1) != 0) {
                    unsigned int vec_load_3[1];
                    {
                        vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_3 + 2 * i_1) + 0);
                    }
                    unsigned int vec_load_1_2[1];
                    {
                        vec_load_1_2[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_3 + K + 2 * i_1) + 0);
                    }
                    float2 cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[0]));
                    float2 cvt_f32_1_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[0]));
                    float e0_47 = approx_exp2((-cvt_f32_48.x) * 1.442695f);
                    float e1_47 = approx_exp2((-cvt_f32_48.y) * 1.442695f);
                    float d0_47 = 1.0f + e0_47;
                    float d1_47 = 1.0f + e1_47;
                    float r0_47 = approx_rcp(d0_47);
                    float r1_47 = approx_rcp(d1_47);
                    float _fma_235;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
                    float fma_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_17) : "f"(r0_47), "f"(_fma_235), "f"(r0_47));
                    float _fma_236;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
                    float fma_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_16) : "f"(r1_47), "f"(_fma_236), "f"(r1_47));
                    float t0_16 = fma_17;
                    float t1_16 = fma_1_16;
                    float d4_8a_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1) : "f"(d0_47), "f"(0.25f));
                    float r4_8a_1 = approx_rcp(d4_8a_1);
                    float _fma_238;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(-d4_8a_1), "f"(r4_8a_1), "f"(1.0f));
                    float _fma_237;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(r4_8a_1), "f"(_fma_238), "f"(r4_8a_1));
                    float big_8a_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1) : "f"(mul_f32_rn_noftz(_fma_237, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_239;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(big_8a_1), "f"(8.388608e+06f));
                    float whole_8a_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1) : "f"(_add_239), "f"(8.388608e+06f));
                    float above_8a_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1) : "f"(-d4_8a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_241;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_241) : "f"(big_8a_1), "f"(whole_8a_1));
                    float _abs_240;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_240) : "f"(_sub_241));
                    float _add_242;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(big_8a_1), "f"(0.5f));
                    float _sub_243;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_243) : "f"(big_8a_1), "f"(0.5f));
                    float rounded_8a_1 = ((_abs_240 == 0.5f) ? ((above_8a_1 > 0.0f) ? _add_242 : _sub_243) : whole_8a_1);
                    float u0_1 = ((d0_47 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                    float d4_8a_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1_1) : "f"(d1_47), "f"(0.25f));
                    float r4_8a_1_1 = approx_rcp(d4_8a_1_1);
                    float _fma_245;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(-d4_8a_1_1), "f"(r4_8a_1_1), "f"(1.0f));
                    float _fma_244;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(r4_8a_1_1), "f"(_fma_245), "f"(r4_8a_1_1));
                    float big_8a_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1_1) : "f"(mul_f32_rn_noftz(_fma_244, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_246;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(big_8a_1_1), "f"(8.388608e+06f));
                    float whole_8a_1_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1_1) : "f"(_add_246), "f"(8.388608e+06f));
                    float above_8a_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1_1) : "f"(-d4_8a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_248;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(big_8a_1_1), "f"(whole_8a_1_1));
                    float _abs_247;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_247) : "f"(_sub_248));
                    float _add_249;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(big_8a_1_1), "f"(0.5f));
                    float _sub_250;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_250) : "f"(big_8a_1_1), "f"(0.5f));
                    float rounded_8a_1_1 = ((_abs_247 == 0.5f) ? ((above_8a_1_1 > 0.0f) ? _add_249 : _sub_250) : whole_8a_1_1);
                    float u1_1 = ((d1_47 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                    float2 _f32x2_mul_a_251 = mul_f32x2_noftz(cvt_f32_48, make_float2(max_noftz(u0_1, 0.0f), max_noftz(u1_1, 0.0f)));
                    float2 mul_47;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_251), "l"(*(const unsigned long long*)&cvt_f32_1_47));
                    unsigned int pair_1 = __as_u32(__float22bfloat162_rn(make_float2(mul_47.x, mul_47.y)));
                    packed_b[0] = ((i_1 == 0) ? pair_1 : packed_b[0]);
                    packed_b[1] = ((i_1 == 1) ? pair_1 : packed_b[1]);
                    packed_b[2] = ((i_1 == 2) ? pair_1 : packed_b[2]);
                    packed_b[3] = ((i_1 == 3) ? pair_1 : packed_b[3]);
                    packed_b[4] = ((i_1 == 4) ? pair_1 : packed_b[4]);
                    packed_b[5] = ((i_1 == 5) ? pair_1 : packed_b[5]);
                    packed_b[6] = ((i_1 == 6) ? pair_1 : packed_b[6]);
                    packed_b[7] = ((i_1 == 7) ? pair_1 : packed_b[7]);
                }
            }
        }
        int band_1 = 0;
        float2 cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float e0_48 = approx_exp2((-cvt_f32_49.x) * 1.442695f);
        float e1_48 = approx_exp2((-cvt_f32_49.y) * 1.442695f);
        float d0_48 = 1.0f + e0_48;
        float d1_48 = 1.0f + e1_48;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float _fma_252;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(-d0_48), "f"(r0_48), "f"(1.0f));
        float fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_18) : "f"(r0_48), "f"(_fma_252), "f"(r0_48));
        float _fma_253;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(-d1_48), "f"(r1_48), "f"(1.0f));
        float fma_1_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_17) : "f"(r1_48), "f"(_fma_253), "f"(r1_48));
        float t0_17 = fma_18;
        float t1_17 = fma_1_17;
        band_1 = band_1 | ((fminf(t0_17, t1_17) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_254 = mul_f32x2_noftz(cvt_f32_49, make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f)));
        float2 mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_254), "l"(*(const unsigned long long*)&cvt_f32_1_48));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_48.x, mul_48.y)));
        float2 cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float e0_49 = approx_exp2((-cvt_f32_50.x) * 1.442695f);
        float e1_49 = approx_exp2((-cvt_f32_50.y) * 1.442695f);
        float d0_49 = 1.0f + e0_49;
        float d1_49 = 1.0f + e1_49;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float _fma_255;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
        float fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_19) : "f"(r0_49), "f"(_fma_255), "f"(r0_49));
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
        float fma_1_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_18) : "f"(r1_49), "f"(_fma_256), "f"(r1_49));
        float t0_18 = fma_19;
        float t1_18 = fma_1_18;
        band_1 = band_1 | ((fminf(t0_18, t1_18) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_257 = mul_f32x2_noftz(cvt_f32_50, make_float2(max_noftz(t0_18, 0.0f), max_noftz(t1_18, 0.0f)));
        float2 mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_257), "l"(*(const unsigned long long*)&cvt_f32_1_49));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_49.x, mul_49.y)));
        float2 cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float e0_50 = approx_exp2((-cvt_f32_51.x) * 1.442695f);
        float e1_50 = approx_exp2((-cvt_f32_51.y) * 1.442695f);
        float d0_50 = 1.0f + e0_50;
        float d1_50 = 1.0f + e1_50;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float _fma_258;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(-d0_50), "f"(r0_50), "f"(1.0f));
        float fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_20) : "f"(r0_50), "f"(_fma_258), "f"(r0_50));
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d1_50), "f"(r1_50), "f"(1.0f));
        float fma_1_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_19) : "f"(r1_50), "f"(_fma_259), "f"(r1_50));
        float t0_19 = fma_20;
        float t1_19 = fma_1_19;
        band_1 = band_1 | ((fminf(t0_19, t1_19) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_260 = mul_f32x2_noftz(cvt_f32_51, make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f)));
        float2 mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_260), "l"(*(const unsigned long long*)&cvt_f32_1_50));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_50.x, mul_50.y)));
        float2 cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float e0_51 = approx_exp2((-cvt_f32_52.x) * 1.442695f);
        float e1_51 = approx_exp2((-cvt_f32_52.y) * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_261;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_21) : "f"(r0_51), "f"(_fma_261), "f"(r0_51));
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float fma_1_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_20) : "f"(r1_51), "f"(_fma_262), "f"(r1_51));
        float t0_20 = fma_21;
        float t1_20 = fma_1_20;
        band_1 = band_1 | ((fminf(t0_20, t1_20) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_263 = mul_f32x2_noftz(cvt_f32_52, make_float2(max_noftz(t0_20, 0.0f), max_noftz(t1_20, 0.0f)));
        float2 mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_263), "l"(*(const unsigned long long*)&cvt_f32_1_51));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_51.x, mul_51.y)));
        float2 cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float e0_52 = approx_exp2((-cvt_f32_53.x) * 1.442695f);
        float e1_52 = approx_exp2((-cvt_f32_53.y) * 1.442695f);
        float d0_52 = 1.0f + e0_52;
        float d1_52 = 1.0f + e1_52;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float _fma_264;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(-d0_52), "f"(r0_52), "f"(1.0f));
        float fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_22) : "f"(r0_52), "f"(_fma_264), "f"(r0_52));
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(-d1_52), "f"(r1_52), "f"(1.0f));
        float fma_1_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_21) : "f"(r1_52), "f"(_fma_265), "f"(r1_52));
        float t0_21 = fma_22;
        float t1_21 = fma_1_21;
        band_1 = band_1 | ((fminf(t0_21, t1_21) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_266 = mul_f32x2_noftz(cvt_f32_53, make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f)));
        float2 mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&cvt_f32_1_52));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_52.x, mul_52.y)));
        float2 cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float e0_53 = approx_exp2((-cvt_f32_54.x) * 1.442695f);
        float e1_53 = approx_exp2((-cvt_f32_54.y) * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_267;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_23) : "f"(r0_53), "f"(_fma_267), "f"(r0_53));
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float fma_1_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_22) : "f"(r1_53), "f"(_fma_268), "f"(r1_53));
        float t0_22 = fma_23;
        float t1_22 = fma_1_22;
        band_1 = band_1 | ((fminf(t0_22, t1_22) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_269 = mul_f32x2_noftz(cvt_f32_54, make_float2(max_noftz(t0_22, 0.0f), max_noftz(t1_22, 0.0f)));
        float2 mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_269), "l"(*(const unsigned long long*)&cvt_f32_1_53));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_53.x, mul_53.y)));
        float2 cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float e0_54 = approx_exp2((-cvt_f32_55.x) * 1.442695f);
        float e1_54 = approx_exp2((-cvt_f32_55.y) * 1.442695f);
        float d0_54 = 1.0f + e0_54;
        float d1_54 = 1.0f + e1_54;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float _fma_270;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(-d0_54), "f"(r0_54), "f"(1.0f));
        float fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_24) : "f"(r0_54), "f"(_fma_270), "f"(r0_54));
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(-d1_54), "f"(r1_54), "f"(1.0f));
        float fma_1_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_23) : "f"(r1_54), "f"(_fma_271), "f"(r1_54));
        float t0_23 = fma_24;
        float t1_23 = fma_1_23;
        band_1 = band_1 | ((fminf(t0_23, t1_23) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_272 = mul_f32x2_noftz(cvt_f32_55, make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f)));
        float2 mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_272), "l"(*(const unsigned long long*)&cvt_f32_1_54));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_54.x, mul_54.y)));
        float2 cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float e0_55 = approx_exp2((-cvt_f32_56.x) * 1.442695f);
        float e1_55 = approx_exp2((-cvt_f32_56.y) * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_273;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_25) : "f"(r0_55), "f"(_fma_273), "f"(r0_55));
        float _fma_274;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float fma_1_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_24) : "f"(r1_55), "f"(_fma_274), "f"(r1_55));
        float t0_24 = fma_25;
        float t1_24 = fma_1_24;
        band_1 = band_1 | ((fminf(t0_24, t1_24) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_275 = mul_f32x2_noftz(cvt_f32_56, make_float2(max_noftz(t0_24, 0.0f), max_noftz(t1_24, 0.0f)));
        float2 mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_275), "l"(*(const unsigned long long*)&cvt_f32_1_55));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_55.x, mul_55.y)));
        if (band_1 != 0) {
            #pragma unroll 1
            for (int i_2 = 0; i_2 < 8; i_2++) {
                if ((band_1 >> i_2 & 1) != 0) {
                    unsigned int vec_load_4[1];
                    {
                        vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_4 + 2 * i_2) + 0);
                    }
                    unsigned int vec_load_1_3[1];
                    {
                        vec_load_1_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_4 + K + 2 * i_2) + 0);
                    }
                    float2 cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[0]));
                    float2 cvt_f32_1_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_3[0]));
                    float e0_56 = approx_exp2((-cvt_f32_57.x) * 1.442695f);
                    float e1_56 = approx_exp2((-cvt_f32_57.y) * 1.442695f);
                    float d0_56 = 1.0f + e0_56;
                    float d1_56 = 1.0f + e1_56;
                    float r0_56 = approx_rcp(d0_56);
                    float r1_56 = approx_rcp(d1_56);
                    float _fma_276;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(-d0_56), "f"(r0_56), "f"(1.0f));
                    float fma_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_26) : "f"(r0_56), "f"(_fma_276), "f"(r0_56));
                    float _fma_277;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(-d1_56), "f"(r1_56), "f"(1.0f));
                    float fma_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_25) : "f"(r1_56), "f"(_fma_277), "f"(r1_56));
                    float t0_25 = fma_26;
                    float t1_25 = fma_1_25;
                    float d4_8a_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2) : "f"(d0_56), "f"(0.25f));
                    float r4_8a_2 = approx_rcp(d4_8a_2);
                    float _fma_279;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(-d4_8a_2), "f"(r4_8a_2), "f"(1.0f));
                    float _fma_278;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(r4_8a_2), "f"(_fma_279), "f"(r4_8a_2));
                    float big_8a_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2) : "f"(mul_f32_rn_noftz(_fma_278, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_280;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(big_8a_2), "f"(8.388608e+06f));
                    float whole_8a_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2) : "f"(_add_280), "f"(8.388608e+06f));
                    float above_8a_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2) : "f"(-d4_8a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_282;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(big_8a_2), "f"(whole_8a_2));
                    float _abs_281;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_281) : "f"(_sub_282));
                    float _add_283;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(big_8a_2), "f"(0.5f));
                    float _sub_284;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(big_8a_2), "f"(0.5f));
                    float rounded_8a_2 = ((_abs_281 == 0.5f) ? ((above_8a_2 > 0.0f) ? _add_283 : _sub_284) : whole_8a_2);
                    float u0_2 = ((d0_56 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2, 7.888609e-31f), 1.7763568e-15f) : t0_25);
                    float d4_8a_2_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2_1) : "f"(d1_56), "f"(0.25f));
                    float r4_8a_2_1 = approx_rcp(d4_8a_2_1);
                    float _fma_286;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(-d4_8a_2_1), "f"(r4_8a_2_1), "f"(1.0f));
                    float _fma_285;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(r4_8a_2_1), "f"(_fma_286), "f"(r4_8a_2_1));
                    float big_8a_2_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2_1) : "f"(mul_f32_rn_noftz(_fma_285, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_287;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(big_8a_2_1), "f"(8.388608e+06f));
                    float whole_8a_2_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2_1) : "f"(_add_287), "f"(8.388608e+06f));
                    float above_8a_2_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2_1) : "f"(-d4_8a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_289;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(big_8a_2_1), "f"(whole_8a_2_1));
                    float _abs_288;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(_sub_289));
                    float _add_290;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(big_8a_2_1), "f"(0.5f));
                    float _sub_291;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(big_8a_2_1), "f"(0.5f));
                    float rounded_8a_2_1 = ((_abs_288 == 0.5f) ? ((above_8a_2_1 > 0.0f) ? _add_290 : _sub_291) : whole_8a_2_1);
                    float u1_2 = ((d1_56 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_25);
                    float2 _f32x2_mul_a_292 = mul_f32x2_noftz(cvt_f32_57, make_float2(max_noftz(u0_2, 0.0f), max_noftz(u1_2, 0.0f)));
                    float2 mul_56;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_292), "l"(*(const unsigned long long*)&cvt_f32_1_56));
                    unsigned int pair_2 = __as_u32(__float22bfloat162_rn(make_float2(mul_56.x, mul_56.y)));
                    packed_c[0] = ((i_2 == 0) ? pair_2 : packed_c[0]);
                    packed_c[1] = ((i_2 == 1) ? pair_2 : packed_c[1]);
                    packed_c[2] = ((i_2 == 2) ? pair_2 : packed_c[2]);
                    packed_c[3] = ((i_2 == 3) ? pair_2 : packed_c[3]);
                    packed_c[4] = ((i_2 == 4) ? pair_2 : packed_c[4]);
                    packed_c[5] = ((i_2 == 5) ? pair_2 : packed_c[5]);
                    packed_c[6] = ((i_2 == 6) ? pair_2 : packed_c[6]);
                    packed_c[7] = ((i_2 == 7) ? pair_2 : packed_c[7]);
                }
            }
        }
        int band_2 = 0;
        float2 cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 cvt_f32_1_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float e0_57 = approx_exp2((-cvt_f32_58.x) * 1.442695f);
        float e1_57 = approx_exp2((-cvt_f32_58.y) * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_293;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_27) : "f"(r0_57), "f"(_fma_293), "f"(r0_57));
        float _fma_294;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float fma_1_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_26) : "f"(r1_57), "f"(_fma_294), "f"(r1_57));
        float t0_26 = fma_27;
        float t1_26 = fma_1_26;
        band_2 = band_2 | ((fminf(t0_26, t1_26) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_295 = mul_f32x2_noftz(cvt_f32_58, make_float2(max_noftz(t0_26, 0.0f), max_noftz(t1_26, 0.0f)));
        float2 mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_295), "l"(*(const unsigned long long*)&cvt_f32_1_57));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_57.x, mul_57.y)));
        float2 cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 cvt_f32_1_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float e0_58 = approx_exp2((-cvt_f32_59.x) * 1.442695f);
        float e1_58 = approx_exp2((-cvt_f32_59.y) * 1.442695f);
        float d0_58 = 1.0f + e0_58;
        float d1_58 = 1.0f + e1_58;
        float r0_58 = approx_rcp(d0_58);
        float r1_58 = approx_rcp(d1_58);
        float _fma_296;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(-d0_58), "f"(r0_58), "f"(1.0f));
        float fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_28) : "f"(r0_58), "f"(_fma_296), "f"(r0_58));
        float _fma_297;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(-d1_58), "f"(r1_58), "f"(1.0f));
        float fma_1_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_27) : "f"(r1_58), "f"(_fma_297), "f"(r1_58));
        float t0_27 = fma_28;
        float t1_27 = fma_1_27;
        band_2 = band_2 | ((fminf(t0_27, t1_27) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_298 = mul_f32x2_noftz(cvt_f32_59, make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f)));
        float2 mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_298), "l"(*(const unsigned long long*)&cvt_f32_1_58));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_58.x, mul_58.y)));
        float2 cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 cvt_f32_1_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float e0_59 = approx_exp2((-cvt_f32_60.x) * 1.442695f);
        float e1_59 = approx_exp2((-cvt_f32_60.y) * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_299;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_29) : "f"(r0_59), "f"(_fma_299), "f"(r0_59));
        float _fma_300;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float fma_1_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_28) : "f"(r1_59), "f"(_fma_300), "f"(r1_59));
        float t0_28 = fma_29;
        float t1_28 = fma_1_28;
        band_2 = band_2 | ((fminf(t0_28, t1_28) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_301 = mul_f32x2_noftz(cvt_f32_60, make_float2(max_noftz(t0_28, 0.0f), max_noftz(t1_28, 0.0f)));
        float2 mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_301), "l"(*(const unsigned long long*)&cvt_f32_1_59));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_59.x, mul_59.y)));
        float2 cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 cvt_f32_1_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float e0_60 = approx_exp2((-cvt_f32_61.x) * 1.442695f);
        float e1_60 = approx_exp2((-cvt_f32_61.y) * 1.442695f);
        float d0_60 = 1.0f + e0_60;
        float d1_60 = 1.0f + e1_60;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float _fma_302;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(-d0_60), "f"(r0_60), "f"(1.0f));
        float fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_30) : "f"(r0_60), "f"(_fma_302), "f"(r0_60));
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d1_60), "f"(r1_60), "f"(1.0f));
        float fma_1_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_29) : "f"(r1_60), "f"(_fma_303), "f"(r1_60));
        float t0_29 = fma_30;
        float t1_29 = fma_1_29;
        band_2 = band_2 | ((fminf(t0_29, t1_29) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_304 = mul_f32x2_noftz(cvt_f32_61, make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f)));
        float2 mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_304), "l"(*(const unsigned long long*)&cvt_f32_1_60));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_60.x, mul_60.y)));
        float2 cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 cvt_f32_1_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float e0_61 = approx_exp2((-cvt_f32_62.x) * 1.442695f);
        float e1_61 = approx_exp2((-cvt_f32_62.y) * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_305;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_31) : "f"(r0_61), "f"(_fma_305), "f"(r0_61));
        float _fma_306;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float fma_1_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_30) : "f"(r1_61), "f"(_fma_306), "f"(r1_61));
        float t0_30 = fma_31;
        float t1_30 = fma_1_30;
        band_2 = band_2 | ((fminf(t0_30, t1_30) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_307 = mul_f32x2_noftz(cvt_f32_62, make_float2(max_noftz(t0_30, 0.0f), max_noftz(t1_30, 0.0f)));
        float2 mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_307), "l"(*(const unsigned long long*)&cvt_f32_1_61));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_61.x, mul_61.y)));
        float2 cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 cvt_f32_1_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float e0_62 = approx_exp2((-cvt_f32_63.x) * 1.442695f);
        float e1_62 = approx_exp2((-cvt_f32_63.y) * 1.442695f);
        float d0_62 = 1.0f + e0_62;
        float d1_62 = 1.0f + e1_62;
        float r0_62 = approx_rcp(d0_62);
        float r1_62 = approx_rcp(d1_62);
        float _fma_308;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(-d0_62), "f"(r0_62), "f"(1.0f));
        float fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_32) : "f"(r0_62), "f"(_fma_308), "f"(r0_62));
        float _fma_309;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(-d1_62), "f"(r1_62), "f"(1.0f));
        float fma_1_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_31) : "f"(r1_62), "f"(_fma_309), "f"(r1_62));
        float t0_31 = fma_32;
        float t1_31 = fma_1_31;
        band_2 = band_2 | ((fminf(t0_31, t1_31) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_310 = mul_f32x2_noftz(cvt_f32_63, make_float2(max_noftz(t0_31, 0.0f), max_noftz(t1_31, 0.0f)));
        float2 mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_310), "l"(*(const unsigned long long*)&cvt_f32_1_62));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_62.x, mul_62.y)));
        float2 cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 cvt_f32_1_63 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float e0_63 = approx_exp2((-cvt_f32_64.x) * 1.442695f);
        float e1_63 = approx_exp2((-cvt_f32_64.y) * 1.442695f);
        float d0_63 = 1.0f + e0_63;
        float d1_63 = 1.0f + e1_63;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float _fma_311;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(-d0_63), "f"(r0_63), "f"(1.0f));
        float fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_33) : "f"(r0_63), "f"(_fma_311), "f"(r0_63));
        float _fma_312;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(-d1_63), "f"(r1_63), "f"(1.0f));
        float fma_1_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_32) : "f"(r1_63), "f"(_fma_312), "f"(r1_63));
        float t0_32 = fma_33;
        float t1_32 = fma_1_32;
        band_2 = band_2 | ((fminf(t0_32, t1_32) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_313 = mul_f32x2_noftz(cvt_f32_64, make_float2(max_noftz(t0_32, 0.0f), max_noftz(t1_32, 0.0f)));
        float2 mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_313), "l"(*(const unsigned long long*)&cvt_f32_1_63));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_63.x, mul_63.y)));
        float2 cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 cvt_f32_1_64 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float e0_64 = approx_exp2((-cvt_f32_65.x) * 1.442695f);
        float e1_64 = approx_exp2((-cvt_f32_65.y) * 1.442695f);
        float d0_64 = 1.0f + e0_64;
        float d1_64 = 1.0f + e1_64;
        float r0_64 = approx_rcp(d0_64);
        float r1_64 = approx_rcp(d1_64);
        float _fma_314;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(-d0_64), "f"(r0_64), "f"(1.0f));
        float fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_34) : "f"(r0_64), "f"(_fma_314), "f"(r0_64));
        float _fma_315;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(-d1_64), "f"(r1_64), "f"(1.0f));
        float fma_1_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_33) : "f"(r1_64), "f"(_fma_315), "f"(r1_64));
        float t0_33 = fma_34;
        float t1_33 = fma_1_33;
        band_2 = band_2 | ((fminf(t0_33, t1_33) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_316 = mul_f32x2_noftz(cvt_f32_65, make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f)));
        float2 mul_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_316), "l"(*(const unsigned long long*)&cvt_f32_1_64));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_64.x, mul_64.y)));
        if (band_2 != 0) {
            #pragma unroll 1
            for (int i_3 = 0; i_3 < 8; i_3++) {
                if ((band_2 >> i_3 & 1) != 0) {
                    unsigned int vec_load_5[1];
                    {
                        vec_load_5[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_5 + 2 * i_3) + 0);
                    }
                    unsigned int vec_load_1_4[1];
                    {
                        vec_load_1_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_5 + K + 2 * i_3) + 0);
                    }
                    float2 cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_5[0]));
                    float2 cvt_f32_1_65 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_4[0]));
                    float e0_65 = approx_exp2((-cvt_f32_66.x) * 1.442695f);
                    float e1_65 = approx_exp2((-cvt_f32_66.y) * 1.442695f);
                    float d0_65 = 1.0f + e0_65;
                    float d1_65 = 1.0f + e1_65;
                    float r0_65 = approx_rcp(d0_65);
                    float r1_65 = approx_rcp(d1_65);
                    float _fma_317;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(-d0_65), "f"(r0_65), "f"(1.0f));
                    float fma_35;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_35) : "f"(r0_65), "f"(_fma_317), "f"(r0_65));
                    float _fma_318;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(-d1_65), "f"(r1_65), "f"(1.0f));
                    float fma_1_34;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_34) : "f"(r1_65), "f"(_fma_318), "f"(r1_65));
                    float t0_34 = fma_35;
                    float t1_34 = fma_1_34;
                    float d4_8a_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3) : "f"(d0_65), "f"(0.25f));
                    float r4_8a_3 = approx_rcp(d4_8a_3);
                    float _fma_320;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(-d4_8a_3), "f"(r4_8a_3), "f"(1.0f));
                    float _fma_319;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(r4_8a_3), "f"(_fma_320), "f"(r4_8a_3));
                    float big_8a_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3) : "f"(mul_f32_rn_noftz(_fma_319, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_321;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_321) : "f"(big_8a_3), "f"(8.388608e+06f));
                    float whole_8a_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3) : "f"(_add_321), "f"(8.388608e+06f));
                    float above_8a_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3) : "f"(-d4_8a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_323;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(big_8a_3), "f"(whole_8a_3));
                    float _abs_322;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_322) : "f"(_sub_323));
                    float _add_324;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(big_8a_3), "f"(0.5f));
                    float _sub_325;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(big_8a_3), "f"(0.5f));
                    float rounded_8a_3 = ((_abs_322 == 0.5f) ? ((above_8a_3 > 0.0f) ? _add_324 : _sub_325) : whole_8a_3);
                    float u0_3 = ((d0_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3, 7.888609e-31f), 1.7763568e-15f) : t0_34);
                    float d4_8a_3_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3_1) : "f"(d1_65), "f"(0.25f));
                    float r4_8a_3_1 = approx_rcp(d4_8a_3_1);
                    float _fma_327;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(-d4_8a_3_1), "f"(r4_8a_3_1), "f"(1.0f));
                    float _fma_326;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(r4_8a_3_1), "f"(_fma_327), "f"(r4_8a_3_1));
                    float big_8a_3_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3_1) : "f"(mul_f32_rn_noftz(_fma_326, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_328;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(big_8a_3_1), "f"(8.388608e+06f));
                    float whole_8a_3_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3_1) : "f"(_add_328), "f"(8.388608e+06f));
                    float above_8a_3_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3_1) : "f"(-d4_8a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_330;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(big_8a_3_1), "f"(whole_8a_3_1));
                    float _abs_329;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_329) : "f"(_sub_330));
                    float _add_331;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(big_8a_3_1), "f"(0.5f));
                    float _sub_332;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_332) : "f"(big_8a_3_1), "f"(0.5f));
                    float rounded_8a_3_1 = ((_abs_329 == 0.5f) ? ((above_8a_3_1 > 0.0f) ? _add_331 : _sub_332) : whole_8a_3_1);
                    float u1_3 = ((d1_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_34);
                    float2 _f32x2_mul_a_333 = mul_f32x2_noftz(cvt_f32_66, make_float2(max_noftz(u0_3, 0.0f), max_noftz(u1_3, 0.0f)));
                    float2 mul_65;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_333), "l"(*(const unsigned long long*)&cvt_f32_1_65));
                    unsigned int pair_3 = __as_u32(__float22bfloat162_rn(make_float2(mul_65.x, mul_65.y)));
                    packed_d[0] = ((i_3 == 0) ? pair_3 : packed_d[0]);
                    packed_d[1] = ((i_3 == 1) ? pair_3 : packed_d[1]);
                    packed_d[2] = ((i_3 == 2) ? pair_3 : packed_d[2]);
                    packed_d[3] = ((i_3 == 3) ? pair_3 : packed_d[3]);
                    packed_d[4] = ((i_3 == 4) ? pair_3 : packed_d[4]);
                    packed_d[5] = ((i_3 == 5) ? pair_3 : packed_d[5]);
                    packed_d[6] = ((i_3 == 6) ? pair_3 : packed_d[6]);
                    packed_d[7] = ((i_3 == 7) ? pair_3 : packed_d[7]);
                }
            }
        }
    }
    unsigned int swizzle = (unsigned int)(row >> 4 & 6 | row & 1);
    unsigned int row_base = tile_addr + (unsigned int)(row * 256);
    unsigned int chunk = (unsigned int)((tid & 7) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk ^ (swizzle ^ 2)) << 4)), "r"(packed_b[0]), "r"(packed_b[1]), "r"(packed_b[2]), "r"(packed_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk + 1 ^ (swizzle ^ 2)) << 4)), "r"(packed_b[4]), "r"(packed_b[5]), "r"(packed_b[6]), "r"(packed_b[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 16384 + ((chunk ^ (swizzle ^ 4)) << 4)), "r"(packed_c[0]), "r"(packed_c[1]), "r"(packed_c[2]), "r"(packed_c[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 16384 + ((chunk + 1 ^ (swizzle ^ 4)) << 4)), "r"(packed_c[4]), "r"(packed_c[5]), "r"(packed_c[6]), "r"(packed_c[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 24576 + ((chunk ^ (swizzle ^ 6)) << 4)), "r"(packed_d[0]), "r"(packed_d[1]), "r"(packed_d[2]), "r"(packed_d[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 24576 + ((chunk + 1 ^ (swizzle ^ 6)) << 4)), "r"(packed_d[4]), "r"(packed_d[5]), "r"(packed_d[6]), "r"(packed_d[7]) : "memory");
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_335 = amax;
    __nv_bfloat162 _hxb_335 = __as_bf16x2(packed[0]);
    unsigned int _hxr_335;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_335) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_335)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_335)));
    __nv_bfloat162 _max_334 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_335);
    amax = _max_334;
    __nv_bfloat162 _hxa_337 = amax;
    __nv_bfloat162 _hxb_337 = __as_bf16x2(packed[1]);
    unsigned int _hxr_337;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_337) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_337)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_337)));
    __nv_bfloat162 _max_336 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_337);
    amax = _max_336;
    __nv_bfloat162 _hxa_339 = amax;
    __nv_bfloat162 _hxb_339 = __as_bf16x2(packed[2]);
    unsigned int _hxr_339;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_339) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_339)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_339)));
    __nv_bfloat162 _max_338 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_339);
    amax = _max_338;
    __nv_bfloat162 _hxa_341 = amax;
    __nv_bfloat162 _hxb_341 = __as_bf16x2(packed[3]);
    unsigned int _hxr_341;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_341) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_341)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_341)));
    __nv_bfloat162 _max_340 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_341);
    amax = _max_340;
    __nv_bfloat162 _hxa_343 = amax;
    __nv_bfloat162 _hxb_343 = __as_bf16x2(packed[4]);
    unsigned int _hxr_343;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_343) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_343)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_343)));
    __nv_bfloat162 _max_342 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_343);
    amax = _max_342;
    __nv_bfloat162 _hxa_345 = amax;
    __nv_bfloat162 _hxb_345 = __as_bf16x2(packed[5]);
    unsigned int _hxr_345;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_345) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_345)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_345)));
    __nv_bfloat162 _max_344 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_345);
    amax = _max_344;
    __nv_bfloat162 _hxa_347 = amax;
    __nv_bfloat162 _hxb_347 = __as_bf16x2(packed[6]);
    unsigned int _hxr_347;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_347) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_347)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_347)));
    __nv_bfloat162 _max_346 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_347);
    amax = _max_346;
    __nv_bfloat162 _hxa_349 = amax;
    __nv_bfloat162 _hxb_349 = __as_bf16x2(packed[7]);
    unsigned int _hxr_349;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_349) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_349)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_349)));
    __nv_bfloat162 _max_348 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_349);
    amax = _max_348;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_351;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_350;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_350) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_351));
    row_words[0] = _mov_b32_pack_u16x2_350;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_353;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_352;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_352) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_353));
    row_words[1] = _mov_b32_pack_u16x2_352;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_355;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_354;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_354) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_355));
    row_words[2] = _mov_b32_pack_u16x2_354;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_357;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_356;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_356) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_357));
    row_words[3] = _mov_b32_pack_u16x2_356;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_359 = amax_0;
    __nv_bfloat162 _hxb_359 = __as_bf16x2(packed_b[0]);
    unsigned int _hxr_359;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_359) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_359)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_359)));
    __nv_bfloat162 _max_358 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_359);
    amax_0 = _max_358;
    __nv_bfloat162 _hxa_361 = amax_0;
    __nv_bfloat162 _hxb_361 = __as_bf16x2(packed_b[1]);
    unsigned int _hxr_361;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_361) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_361)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_361)));
    __nv_bfloat162 _max_360 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_361);
    amax_0 = _max_360;
    __nv_bfloat162 _hxa_363 = amax_0;
    __nv_bfloat162 _hxb_363 = __as_bf16x2(packed_b[2]);
    unsigned int _hxr_363;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_363) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_363)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_363)));
    __nv_bfloat162 _max_362 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_363);
    amax_0 = _max_362;
    __nv_bfloat162 _hxa_365 = amax_0;
    __nv_bfloat162 _hxb_365 = __as_bf16x2(packed_b[3]);
    unsigned int _hxr_365;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_365) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_365)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_365)));
    __nv_bfloat162 _max_364 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_365);
    amax_0 = _max_364;
    __nv_bfloat162 _hxa_367 = amax_0;
    __nv_bfloat162 _hxb_367 = __as_bf16x2(packed_b[4]);
    unsigned int _hxr_367;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_367) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_367)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_367)));
    __nv_bfloat162 _max_366 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_367);
    amax_0 = _max_366;
    __nv_bfloat162 _hxa_369 = amax_0;
    __nv_bfloat162 _hxb_369 = __as_bf16x2(packed_b[5]);
    unsigned int _hxr_369;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_369) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_369)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_369)));
    __nv_bfloat162 _max_368 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_369);
    amax_0 = _max_368;
    __nv_bfloat162 _hxa_371 = amax_0;
    __nv_bfloat162 _hxb_371 = __as_bf16x2(packed_b[6]);
    unsigned int _hxr_371;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_371) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_371)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_371)));
    __nv_bfloat162 _max_370 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_371);
    amax_0 = _max_370;
    __nv_bfloat162 _hxa_373 = amax_0;
    __nv_bfloat162 _hxb_373 = __as_bf16x2(packed_b[7]);
    unsigned int _hxr_373;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_373) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_373)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_373)));
    __nv_bfloat162 _max_372 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_373);
    amax_0 = _max_372;
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
    int row_scale_index_0 = ((grow_b >> 7) * (K >> 7) + (scale_col_0 >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col_0 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int row_words_0[4];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(packed_b[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(packed_b[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_375;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_375) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_375) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_374;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_374) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_375));
    row_words_0[0] = _mov_b32_pack_u16x2_374;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(packed_b[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(packed_b[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_377;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_377) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_377) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_376;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_376) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_377));
    row_words_0[1] = _mov_b32_pack_u16x2_376;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(packed_b[4]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(packed_b[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_379;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_379) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_379) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_378;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_378) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_379));
    row_words_0[2] = _mov_b32_pack_u16x2_378;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(packed_b[6]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(packed_b[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_381;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_381) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_381) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_380;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_380) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_381));
    row_words_0[3] = _mov_b32_pack_u16x2_380;
    reinterpret_cast<int4*>(row_output + (grow_b * K + col))[0] = reinterpret_cast<int4*>(row_words_0 + 0)[0];
    __nv_bfloat162 amax_16 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_383 = amax_16;
    __nv_bfloat162 _hxb_383 = __as_bf16x2(packed_c[0]);
    unsigned int _hxr_383;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_383) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_383)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_383)));
    __nv_bfloat162 _max_382 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_383);
    amax_16 = _max_382;
    __nv_bfloat162 _hxa_385 = amax_16;
    __nv_bfloat162 _hxb_385 = __as_bf16x2(packed_c[1]);
    unsigned int _hxr_385;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_385) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_385)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_385)));
    __nv_bfloat162 _max_384 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_385);
    amax_16 = _max_384;
    __nv_bfloat162 _hxa_387 = amax_16;
    __nv_bfloat162 _hxb_387 = __as_bf16x2(packed_c[2]);
    unsigned int _hxr_387;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_387) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_387)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_387)));
    __nv_bfloat162 _max_386 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_387);
    amax_16 = _max_386;
    __nv_bfloat162 _hxa_389 = amax_16;
    __nv_bfloat162 _hxb_389 = __as_bf16x2(packed_c[3]);
    unsigned int _hxr_389;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_389) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_389)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_389)));
    __nv_bfloat162 _max_388 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_389);
    amax_16 = _max_388;
    __nv_bfloat162 _hxa_391 = amax_16;
    __nv_bfloat162 _hxb_391 = __as_bf16x2(packed_c[4]);
    unsigned int _hxr_391;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_391) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_391)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_391)));
    __nv_bfloat162 _max_390 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_391);
    amax_16 = _max_390;
    __nv_bfloat162 _hxa_393 = amax_16;
    __nv_bfloat162 _hxb_393 = __as_bf16x2(packed_c[5]);
    unsigned int _hxr_393;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_393) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_393)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_393)));
    __nv_bfloat162 _max_392 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_393);
    amax_16 = _max_392;
    __nv_bfloat162 _hxa_395 = amax_16;
    __nv_bfloat162 _hxb_395 = __as_bf16x2(packed_c[6]);
    unsigned int _hxr_395;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_395) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_395)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_395)));
    __nv_bfloat162 _max_394 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_395);
    amax_16 = _max_394;
    __nv_bfloat162 _hxa_397 = amax_16;
    __nv_bfloat162 _hxb_397 = __as_bf16x2(packed_c[7]);
    unsigned int _hxr_397;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_397) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_397)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_397)));
    __nv_bfloat162 _max_396 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_397);
    amax_16 = _max_396;
    unsigned int mag_17 = __as_u32(amax_16) & 2147450879;
    unsigned int peer_18 = __shfl_xor_sync(0xFFFFFFFF, mag_17, 1);
    __nv_bfloat162 both_19 = __hmax2_nan(__as_bf16x2(mag_17), __as_bf16x2(peer_18));
    unsigned int folded_20 = __as_u32(__hmax2_nan(both_19, __as_bf16x2(__as_u32(both_19) >> 16)));
    unsigned int amax_bits_1 = (folded_20 & 65535) << 16;
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
    int scale_col_26 = bx * 4 + blk;
    int row_scale_index_27 = ((grow_c >> 7) * (K >> 7) + (scale_col_26 >> 2)) * 512 + (grow_c & 31) * 16 + (grow_c >> 5 & 3) * 4 + (scale_col_26 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_27) + (0)) = (unsigned char)(scale_1);
    }
    unsigned int row_words_28[4];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(packed_c[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(packed_c[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_399;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_398;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_398) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_399));
    row_words_28[0] = _mov_b32_pack_u16x2_398;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(packed_c[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(packed_c[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_401;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_401) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_401) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_400;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_400) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_401));
    row_words_28[1] = _mov_b32_pack_u16x2_400;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(packed_c[4]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(packed_c[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_403;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_402;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_402) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_403));
    row_words_28[2] = _mov_b32_pack_u16x2_402;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(packed_c[6]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(packed_c[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_405;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_404;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_404) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_405));
    row_words_28[3] = _mov_b32_pack_u16x2_404;
    reinterpret_cast<int4*>(row_output + (grow_c * K + col))[0] = reinterpret_cast<int4*>(row_words_28 + 0)[0];
    __nv_bfloat162 amax_29 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_407 = amax_29;
    __nv_bfloat162 _hxb_407 = __as_bf16x2(packed_d[0]);
    unsigned int _hxr_407;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_407) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_407)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_407)));
    __nv_bfloat162 _max_406 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_407);
    amax_29 = _max_406;
    __nv_bfloat162 _hxa_409 = amax_29;
    __nv_bfloat162 _hxb_409 = __as_bf16x2(packed_d[1]);
    unsigned int _hxr_409;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_409) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_409)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_409)));
    __nv_bfloat162 _max_408 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_409);
    amax_29 = _max_408;
    __nv_bfloat162 _hxa_411 = amax_29;
    __nv_bfloat162 _hxb_411 = __as_bf16x2(packed_d[2]);
    unsigned int _hxr_411;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_411) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_411)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_411)));
    __nv_bfloat162 _max_410 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_411);
    amax_29 = _max_410;
    __nv_bfloat162 _hxa_413 = amax_29;
    __nv_bfloat162 _hxb_413 = __as_bf16x2(packed_d[3]);
    unsigned int _hxr_413;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_413) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_413)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_413)));
    __nv_bfloat162 _max_412 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_413);
    amax_29 = _max_412;
    __nv_bfloat162 _hxa_415 = amax_29;
    __nv_bfloat162 _hxb_415 = __as_bf16x2(packed_d[4]);
    unsigned int _hxr_415;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_415) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_415)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_415)));
    __nv_bfloat162 _max_414 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_415);
    amax_29 = _max_414;
    __nv_bfloat162 _hxa_417 = amax_29;
    __nv_bfloat162 _hxb_417 = __as_bf16x2(packed_d[5]);
    unsigned int _hxr_417;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_417) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_417)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_417)));
    __nv_bfloat162 _max_416 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_417);
    amax_29 = _max_416;
    __nv_bfloat162 _hxa_419 = amax_29;
    __nv_bfloat162 _hxb_419 = __as_bf16x2(packed_d[6]);
    unsigned int _hxr_419;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_419) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_419)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_419)));
    __nv_bfloat162 _max_418 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_419);
    amax_29 = _max_418;
    __nv_bfloat162 _hxa_421 = amax_29;
    __nv_bfloat162 _hxb_421 = __as_bf16x2(packed_d[7]);
    unsigned int _hxr_421;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_421) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_421)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_421)));
    __nv_bfloat162 _max_420 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_421);
    amax_29 = _max_420;
    unsigned int mag_30 = __as_u32(amax_29) & 2147450879;
    unsigned int peer_31 = __shfl_xor_sync(0xFFFFFFFF, mag_30, 1);
    __nv_bfloat162 both_32 = __hmax2_nan(__as_bf16x2(mag_30), __as_bf16x2(peer_31));
    unsigned int folded_33 = __as_u32(__hmax2_nan(both_32, __as_bf16x2(__as_u32(both_32) >> 16)));
    unsigned int amax_bits_2 = (folded_33 & 65535) << 16;
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
    int scale_col_39 = bx * 4 + blk;
    int row_scale_index_40 = ((grow_d >> 7) * (K >> 7) + (scale_col_39 >> 2)) * 512 + (grow_d & 31) * 16 + (grow_d >> 5 & 3) * 4 + (scale_col_39 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_40) + (0)) = (unsigned char)(scale_3);
    }
    unsigned int row_words_41[4];
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(packed_d[0]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(packed_d[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_423;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_422;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_422) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_423));
    row_words_41[0] = _mov_b32_pack_u16x2_422;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(packed_d[2]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(packed_d[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_425;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_424;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_424) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_425));
    row_words_41[1] = _mov_b32_pack_u16x2_424;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(packed_d[4]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(packed_d[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_427;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_426;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_426) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_427));
    row_words_41[2] = _mov_b32_pack_u16x2_426;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(packed_d[6]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(packed_d[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_429;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_428;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_428) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_429));
    row_words_41[3] = _mov_b32_pack_u16x2_428;
    reinterpret_cast<int4*>(row_output + (grow_d * K + col))[0] = reinterpret_cast<int4*>(row_words_41 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 32 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 4 & 6 | srow & 1);
    unsigned int cols[32];
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_0 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_0 + 6144)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src0_1_1 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src0_1_1 + 6144)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_431 = amax_a;
    __nv_bfloat162 _hxb_431 = __as_bf16x2(cols[0]);
    unsigned int _hxr_431;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_431) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_431)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_431)));
    __nv_bfloat162 _max_430 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_431);
    amax_a = _max_430;
    __nv_bfloat162 _hxa_433 = amax_b;
    __nv_bfloat162 _hxb_433 = __as_bf16x2(cols[16]);
    unsigned int _hxr_433;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_433) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_433)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_433)));
    __nv_bfloat162 _max_432 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_433);
    amax_b = _max_432;
    __nv_bfloat162 _hxa_435 = amax_a;
    __nv_bfloat162 _hxb_435 = __as_bf16x2(cols[1]);
    unsigned int _hxr_435;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_435) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_435)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_435)));
    __nv_bfloat162 _max_434 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_435);
    amax_a = _max_434;
    __nv_bfloat162 _hxa_437 = amax_b;
    __nv_bfloat162 _hxb_437 = __as_bf16x2(cols[17]);
    unsigned int _hxr_437;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_437) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_437)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_437)));
    __nv_bfloat162 _max_436 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_437);
    amax_b = _max_436;
    __nv_bfloat162 _hxa_439 = amax_a;
    __nv_bfloat162 _hxb_439 = __as_bf16x2(cols[2]);
    unsigned int _hxr_439;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_439) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_439)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_439)));
    __nv_bfloat162 _max_438 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_439);
    amax_a = _max_438;
    __nv_bfloat162 _hxa_441 = amax_b;
    __nv_bfloat162 _hxb_441 = __as_bf16x2(cols[18]);
    unsigned int _hxr_441;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_441) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_441)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_441)));
    __nv_bfloat162 _max_440 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_441);
    amax_b = _max_440;
    __nv_bfloat162 _hxa_443 = amax_a;
    __nv_bfloat162 _hxb_443 = __as_bf16x2(cols[3]);
    unsigned int _hxr_443;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_443) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_443)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_443)));
    __nv_bfloat162 _max_442 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_443);
    amax_a = _max_442;
    __nv_bfloat162 _hxa_445 = amax_b;
    __nv_bfloat162 _hxb_445 = __as_bf16x2(cols[19]);
    unsigned int _hxr_445;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_445) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_445)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_445)));
    __nv_bfloat162 _max_444 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_445);
    amax_b = _max_444;
    __nv_bfloat162 _hxa_447 = amax_a;
    __nv_bfloat162 _hxb_447 = __as_bf16x2(cols[4]);
    unsigned int _hxr_447;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_447) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_447)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_447)));
    __nv_bfloat162 _max_446 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_447);
    amax_a = _max_446;
    __nv_bfloat162 _hxa_449 = amax_b;
    __nv_bfloat162 _hxb_449 = __as_bf16x2(cols[20]);
    unsigned int _hxr_449;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_449) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_449)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_449)));
    __nv_bfloat162 _max_448 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_449);
    amax_b = _max_448;
    __nv_bfloat162 _hxa_451 = amax_a;
    __nv_bfloat162 _hxb_451 = __as_bf16x2(cols[5]);
    unsigned int _hxr_451;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_451) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_451)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_451)));
    __nv_bfloat162 _max_450 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_451);
    amax_a = _max_450;
    __nv_bfloat162 _hxa_453 = amax_b;
    __nv_bfloat162 _hxb_453 = __as_bf16x2(cols[21]);
    unsigned int _hxr_453;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_453) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_453)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_453)));
    __nv_bfloat162 _max_452 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_453);
    amax_b = _max_452;
    __nv_bfloat162 _hxa_455 = amax_a;
    __nv_bfloat162 _hxb_455 = __as_bf16x2(cols[6]);
    unsigned int _hxr_455;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_455) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_455)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_455)));
    __nv_bfloat162 _max_454 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_455);
    amax_a = _max_454;
    __nv_bfloat162 _hxa_457 = amax_b;
    __nv_bfloat162 _hxb_457 = __as_bf16x2(cols[22]);
    unsigned int _hxr_457;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_457) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_457)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_457)));
    __nv_bfloat162 _max_456 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_457);
    amax_b = _max_456;
    __nv_bfloat162 _hxa_459 = amax_a;
    __nv_bfloat162 _hxb_459 = __as_bf16x2(cols[7]);
    unsigned int _hxr_459;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_459) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_459)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_459)));
    __nv_bfloat162 _max_458 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_459);
    amax_a = _max_458;
    __nv_bfloat162 _hxa_461 = amax_b;
    __nv_bfloat162 _hxb_461 = __as_bf16x2(cols[23]);
    unsigned int _hxr_461;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_461) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_461)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_461)));
    __nv_bfloat162 _max_460 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_461);
    amax_b = _max_460;
    __nv_bfloat162 _hxa_463 = amax_a;
    __nv_bfloat162 _hxb_463 = __as_bf16x2(cols[8]);
    unsigned int _hxr_463;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_463) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_463)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_463)));
    __nv_bfloat162 _max_462 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_463);
    amax_a = _max_462;
    __nv_bfloat162 _hxa_465 = amax_b;
    __nv_bfloat162 _hxb_465 = __as_bf16x2(cols[24]);
    unsigned int _hxr_465;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_465) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_465)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_465)));
    __nv_bfloat162 _max_464 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_465);
    amax_b = _max_464;
    __nv_bfloat162 _hxa_467 = amax_a;
    __nv_bfloat162 _hxb_467 = __as_bf16x2(cols[9]);
    unsigned int _hxr_467;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_467) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_467)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_467)));
    __nv_bfloat162 _max_466 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_467);
    amax_a = _max_466;
    __nv_bfloat162 _hxa_469 = amax_b;
    __nv_bfloat162 _hxb_469 = __as_bf16x2(cols[25]);
    unsigned int _hxr_469;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_469) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_469)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_469)));
    __nv_bfloat162 _max_468 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_469);
    amax_b = _max_468;
    __nv_bfloat162 _hxa_471 = amax_a;
    __nv_bfloat162 _hxb_471 = __as_bf16x2(cols[10]);
    unsigned int _hxr_471;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_471) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_471)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_471)));
    __nv_bfloat162 _max_470 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_471);
    amax_a = _max_470;
    __nv_bfloat162 _hxa_473 = amax_b;
    __nv_bfloat162 _hxb_473 = __as_bf16x2(cols[26]);
    unsigned int _hxr_473;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_473) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_473)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_473)));
    __nv_bfloat162 _max_472 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_473);
    amax_b = _max_472;
    __nv_bfloat162 _hxa_475 = amax_a;
    __nv_bfloat162 _hxb_475 = __as_bf16x2(cols[11]);
    unsigned int _hxr_475;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_475) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_475)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_475)));
    __nv_bfloat162 _max_474 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_475);
    amax_a = _max_474;
    __nv_bfloat162 _hxa_477 = amax_b;
    __nv_bfloat162 _hxb_477 = __as_bf16x2(cols[27]);
    unsigned int _hxr_477;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_477) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_477)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_477)));
    __nv_bfloat162 _max_476 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_477);
    amax_b = _max_476;
    __nv_bfloat162 _hxa_479 = amax_a;
    __nv_bfloat162 _hxb_479 = __as_bf16x2(cols[12]);
    unsigned int _hxr_479;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_479) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_479)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_479)));
    __nv_bfloat162 _max_478 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_479);
    amax_a = _max_478;
    __nv_bfloat162 _hxa_481 = amax_b;
    __nv_bfloat162 _hxb_481 = __as_bf16x2(cols[28]);
    unsigned int _hxr_481;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_481) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_481)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_481)));
    __nv_bfloat162 _max_480 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_481);
    amax_b = _max_480;
    __nv_bfloat162 _hxa_483 = amax_a;
    __nv_bfloat162 _hxb_483 = __as_bf16x2(cols[13]);
    unsigned int _hxr_483;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_483) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_483)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_483)));
    __nv_bfloat162 _max_482 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_483);
    amax_a = _max_482;
    __nv_bfloat162 _hxa_485 = amax_b;
    __nv_bfloat162 _hxb_485 = __as_bf16x2(cols[29]);
    unsigned int _hxr_485;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_485) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_485)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_485)));
    __nv_bfloat162 _max_484 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_485);
    amax_b = _max_484;
    __nv_bfloat162 _hxa_487 = amax_a;
    __nv_bfloat162 _hxb_487 = __as_bf16x2(cols[14]);
    unsigned int _hxr_487;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_487) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_487)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_487)));
    __nv_bfloat162 _max_486 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_487);
    amax_a = _max_486;
    __nv_bfloat162 _hxa_489 = amax_b;
    __nv_bfloat162 _hxb_489 = __as_bf16x2(cols[30]);
    unsigned int _hxr_489;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_489) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_489)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_489)));
    __nv_bfloat162 _max_488 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_489);
    amax_b = _max_488;
    __nv_bfloat162 _hxa_491 = amax_a;
    __nv_bfloat162 _hxb_491 = __as_bf16x2(cols[15]);
    unsigned int _hxr_491;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_491) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_491)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_491)));
    __nv_bfloat162 _max_490 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_491);
    amax_a = _max_490;
    __nv_bfloat162 _hxa_493 = amax_b;
    __nv_bfloat162 _hxb_493 = __as_bf16x2(cols[31]);
    unsigned int _hxr_493;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_493) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_493)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_493)));
    __nv_bfloat162 _max_492 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_493);
    amax_b = _max_492;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits_3 = (col_max_bits & 65535) << 16;
    int scale_i32_3 = (int)(amax_bits_3 + 2031616 >> 23) - 8;
    if (scale_i32_3 < 0) {
        scale_i32_3 = 0;
    }
    if ((amax_bits_3 & 2139095040) == 2139095040) {
        scale_i32_3 = 255;
    }
    unsigned int scale_4 = (unsigned int)scale_i32_3;
    unsigned int inv_3 = 254 - scale_4 << 7;
    if (scale_4 == 255) {
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
    unsigned int scale_6 = (unsigned int)scale_i32_4;
    unsigned int inv_4 = 254 - scale_6 << 7;
    if (scale_6 == 255) {
        inv_4 = 32704;
    }
    unsigned int inv2_5 = inv_4 | inv_4 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_495;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_494;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_494) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_495));
    words_a[0] = _mov_b32_pack_u16x2_494;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(cols[16]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols[17]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_497;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_496;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_496) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_497));
    words_b[0] = _mov_b32_pack_u16x2_496;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_499;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_498;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_498) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_499));
    words_a[1] = _mov_b32_pack_u16x2_498;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols[18]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols[19]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_501;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_500;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_500) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_501));
    words_b[1] = _mov_b32_pack_u16x2_500;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_503;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_502;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_502) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_503));
    words_a[2] = _mov_b32_pack_u16x2_502;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols[20]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols[21]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_505;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_504;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_504) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_505));
    words_b[2] = _mov_b32_pack_u16x2_504;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_507;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_506;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_506) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_507));
    words_a[3] = _mov_b32_pack_u16x2_506;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols[22]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols[23]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_509;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_508;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_508) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_509));
    words_b[3] = _mov_b32_pack_u16x2_508;
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols[8]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols[9]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_511;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_510;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_510) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_511));
    words_a[4] = _mov_b32_pack_u16x2_510;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols[24]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols[25]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_513;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_512;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_512) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_513));
    words_b[4] = _mov_b32_pack_u16x2_512;
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols[10]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols[11]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_515;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_514;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_514) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_515));
    words_a[5] = _mov_b32_pack_u16x2_514;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols[26]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols[27]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_517;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_516;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_516) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_517));
    words_b[5] = _mov_b32_pack_u16x2_516;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols[12]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols[13]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_519;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_518;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_518) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_519));
    words_a[6] = _mov_b32_pack_u16x2_518;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols[28]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols[29]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_521;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_520;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_520) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_521));
    words_b[6] = _mov_b32_pack_u16x2_520;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols[14]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols[15]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_523;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_522;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_522) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_523));
    words_a[7] = _mov_b32_pack_u16x2_522;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols[30]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols[31]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_525;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_524;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_524) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_525));
    words_b[7] = _mov_b32_pack_u16x2_524;
    {
        const unsigned* _raw_stv8_526 = reinterpret_cast<const unsigned*>(words_a + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a * M + out_row))), "r"(_raw_stv8_526[0]), "r"(_raw_stv8_526[1]), "r"(_raw_stv8_526[2]), "r"(_raw_stv8_526[3]), "r"(_raw_stv8_526[4]), "r"(_raw_stv8_526[5]), "r"(_raw_stv8_526[6]), "r"(_raw_stv8_526[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_527 = reinterpret_cast<const unsigned*>(words_b + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b * M + out_row))), "r"(_raw_stv8_527[0]), "r"(_raw_stv8_527[1]), "r"(_raw_stv8_527[2]), "r"(_raw_stv8_527[3]), "r"(_raw_stv8_527[4]), "r"(_raw_stv8_527[5]), "r"(_raw_stv8_527[6]), "r"(_raw_stv8_527[7]) : "memory");
    }
    int row_tiles = M >> 7;
    int rblk = by * 4 + (lane_1 & 3);
    int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
    int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_4);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_6);
}

} // extern "C"
