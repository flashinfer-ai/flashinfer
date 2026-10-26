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
    int gate_index_0 = grow_b * (2 * K) + col;
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_0 + (0))) : "memory");
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_0 + K) + (0))) : "memory");
        _vec_load_3[0 + 0] = (uint32_t)_iv_3_0;
        _vec_load_3[0 + 1] = (uint32_t)_iv_3_1;
        _vec_load_3[0 + 2] = (uint32_t)_iv_3_2;
        _vec_load_3[0 + 3] = (uint32_t)_iv_3_3;
        _vec_load_3[0 + 4] = (uint32_t)_iv_3_4;
        _vec_load_3[0 + 5] = (uint32_t)_iv_3_5;
        _vec_load_3[0 + 6] = (uint32_t)_iv_3_6;
        _vec_load_3[0 + 7] = (uint32_t)_iv_3_7;
    }
    int gate_index_1 = grow_c * (2 * K) + col;
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + gate_index_1 + (0))) : "memory");
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
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(gated_input + (gate_index_1 + K) + (0))) : "memory");
        _vec_load_5[0 + 0] = (uint32_t)_iv_5_0;
        _vec_load_5[0 + 1] = (uint32_t)_iv_5_1;
        _vec_load_5[0 + 2] = (uint32_t)_iv_5_2;
        _vec_load_5[0 + 3] = (uint32_t)_iv_5_3;
        _vec_load_5[0 + 4] = (uint32_t)_iv_5_4;
        _vec_load_5[0 + 5] = (uint32_t)_iv_5_5;
        _vec_load_5[0 + 6] = (uint32_t)_iv_5_6;
        _vec_load_5[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_2 = grow_d * (2 * K) + col;
    unsigned int _vec_load_6[8];
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
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_2 + (0))) : "memory");
        _vec_load_6[0 + 0] = (uint32_t)_iv_6_0;
        _vec_load_6[0 + 1] = (uint32_t)_iv_6_1;
        _vec_load_6[0 + 2] = (uint32_t)_iv_6_2;
        _vec_load_6[0 + 3] = (uint32_t)_iv_6_3;
        _vec_load_6[0 + 4] = (uint32_t)_iv_6_4;
        _vec_load_6[0 + 5] = (uint32_t)_iv_6_5;
        _vec_load_6[0 + 6] = (uint32_t)_iv_6_6;
        _vec_load_6[0 + 7] = (uint32_t)_iv_6_7;
    }
    unsigned int _vec_load_7[8];
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
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_2 + K) + (0))) : "memory");
        _vec_load_7[0 + 0] = (uint32_t)_iv_7_0;
        _vec_load_7[0 + 1] = (uint32_t)_iv_7_1;
        _vec_load_7[0 + 2] = (uint32_t)_iv_7_2;
        _vec_load_7[0 + 3] = (uint32_t)_iv_7_3;
        _vec_load_7[0 + 4] = (uint32_t)_iv_7_4;
        _vec_load_7[0 + 5] = (uint32_t)_iv_7_5;
        _vec_load_7[0 + 6] = (uint32_t)_iv_7_6;
        _vec_load_7[0 + 7] = (uint32_t)_iv_7_7;
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
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[0]));
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
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_9;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_8 = __all_sync(0xFFFFFFFF, _min_9 > -87.0f);
    if (_warp_vote_8 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_0 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_0 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
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
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(_cvt_f32_1, make_float2(_fma_10, _fma_12));
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_cvt_f32_2));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y)));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_1 = approx_exp2((-_cvt_f32_3.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_3.y) * 1.442695f);
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
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(_cvt_f32_3, make_float2(_fma_15, _fma_17));
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_cvt_f32_4));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y)));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_2 = approx_exp2((-_cvt_f32_5.x) * 1.442695f);
        float e1_2 = approx_exp2((-_cvt_f32_5.y) * 1.442695f);
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
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(_cvt_f32_5, make_float2(_fma_20, _fma_22));
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_cvt_f32_6));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y)));
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_3 = approx_exp2((-_cvt_f32_7.x) * 1.442695f);
        float e1_3 = approx_exp2((-_cvt_f32_7.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r0_3), "f"(_fma_26), "f"(r0_3));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_3), "f"(_fma_28), "f"(r1_3));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(_cvt_f32_7, make_float2(_fma_25, _fma_27));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_cvt_f32_8));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y)));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_4 = approx_exp2((-_cvt_f32_9.x) * 1.442695f);
        float e1_4 = approx_exp2((-_cvt_f32_9.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_4), "f"(_fma_31), "f"(r0_4));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r1_4), "f"(_fma_33), "f"(r1_4));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(_cvt_f32_9, make_float2(_fma_30, _fma_32));
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_cvt_f32_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y)));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_5 = approx_exp2((-_cvt_f32_11.x) * 1.442695f);
        float e1_5 = approx_exp2((-_cvt_f32_11.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r0_5), "f"(_fma_36), "f"(r0_5));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r1_5), "f"(_fma_38), "f"(r1_5));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(_cvt_f32_11, make_float2(_fma_35, _fma_37));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_cvt_f32_12));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y)));
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_6 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_6 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r0_6), "f"(_fma_41), "f"(r0_6));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r1_6), "f"(_fma_43), "f"(r1_6));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(_cvt_f32_13, make_float2(_fma_40, _fma_42));
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_cvt_f32_14));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y)));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_7 = approx_exp2((-_cvt_f32_15.x) * 1.442695f);
        float e1_7 = approx_exp2((-_cvt_f32_15.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(r0_7), "f"(_fma_46), "f"(r0_7));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r1_7), "f"(_fma_48), "f"(r1_7));
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(_cvt_f32_15, make_float2(_fma_45, _fma_47));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&_cvt_f32_16));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y)));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0_8 = approx_exp2((-_cvt_f32_17.x) * 1.442695f);
        float e1_8 = approx_exp2((-_cvt_f32_17.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(r0_8), "f"(_fma_51), "f"(r0_8));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(r1_8), "f"(_fma_53), "f"(r1_8));
        float2 _f32x2_mul_a_54 = mul_f32x2_noftz(_cvt_f32_17, make_float2(_fma_50, _fma_52));
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_cvt_f32_18));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y)));
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_9 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(r0_9), "f"(_fma_56), "f"(r0_9));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(r1_9), "f"(_fma_58), "f"(r1_9));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(_cvt_f32_19, make_float2(_fma_55, _fma_57));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_cvt_f32_20));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y)));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_10 = approx_exp2((-_cvt_f32_21.x) * 1.442695f);
        float e1_10 = approx_exp2((-_cvt_f32_21.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(r0_10), "f"(_fma_61), "f"(r0_10));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(r1_10), "f"(_fma_63), "f"(r1_10));
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(_cvt_f32_21, make_float2(_fma_60, _fma_62));
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_cvt_f32_22));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y)));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_11 = approx_exp2((-_cvt_f32_23.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_23.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(r0_11), "f"(_fma_66), "f"(r0_11));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(r1_11), "f"(_fma_68), "f"(r1_11));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(_cvt_f32_23, make_float2(_fma_65, _fma_67));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_cvt_f32_24));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y)));
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_12 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_12 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(r0_12), "f"(_fma_71), "f"(r0_12));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(r1_12), "f"(_fma_73), "f"(r1_12));
        float2 _f32x2_mul_a_74 = mul_f32x2_noftz(_cvt_f32_25, make_float2(_fma_70, _fma_72));
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_cvt_f32_26));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y)));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_13 = approx_exp2((-_cvt_f32_27.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_27.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(r0_13), "f"(_fma_76), "f"(r0_13));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(r1_13), "f"(_fma_78), "f"(r1_13));
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(_cvt_f32_27, make_float2(_fma_75, _fma_77));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_cvt_f32_28));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y)));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_14 = approx_exp2((-_cvt_f32_29.x) * 1.442695f);
        float e1_14 = approx_exp2((-_cvt_f32_29.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(r0_14), "f"(_fma_81), "f"(r0_14));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(r1_14), "f"(_fma_83), "f"(r1_14));
        float2 _f32x2_mul_a_84 = mul_f32x2_noftz(_cvt_f32_29, make_float2(_fma_80, _fma_82));
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_cvt_f32_30));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y)));
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_15 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(r0_15), "f"(_fma_86), "f"(r0_15));
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(r1_15), "f"(_fma_88), "f"(r1_15));
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(_cvt_f32_31, make_float2(_fma_85, _fma_87));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_cvt_f32_32));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y)));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_16 = approx_exp2((-_cvt_f32_33.x) * 1.442695f);
        float e1_16 = approx_exp2((-_cvt_f32_33.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(r0_16), "f"(_fma_91), "f"(r0_16));
        float _fma_93;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(r1_16), "f"(_fma_93), "f"(r1_16));
        float2 _f32x2_mul_a_94 = mul_f32x2_noftz(_cvt_f32_33, make_float2(_fma_90, _fma_92));
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&_cvt_f32_34));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y)));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_17 = approx_exp2((-_cvt_f32_35.x) * 1.442695f);
        float e1_17 = approx_exp2((-_cvt_f32_35.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(r0_17), "f"(_fma_96), "f"(r0_17));
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(r1_17), "f"(_fma_98), "f"(r1_17));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(_cvt_f32_35, make_float2(_fma_95, _fma_97));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_cvt_f32_36));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y)));
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_18 = approx_exp2((-_cvt_f32_37.x) * 1.442695f);
        float e1_18 = approx_exp2((-_cvt_f32_37.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(r0_18), "f"(_fma_101), "f"(r0_18));
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(r1_18), "f"(_fma_103), "f"(r1_18));
        float2 _f32x2_mul_a_104 = mul_f32x2_noftz(_cvt_f32_37, make_float2(_fma_100, _fma_102));
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_cvt_f32_38));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y)));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_19 = approx_exp2((-_cvt_f32_39.x) * 1.442695f);
        float e1_19 = approx_exp2((-_cvt_f32_39.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(r0_19), "f"(_fma_106), "f"(r0_19));
        float _fma_108;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(r1_19), "f"(_fma_108), "f"(r1_19));
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(_cvt_f32_39, make_float2(_fma_105, _fma_107));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_cvt_f32_40));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y)));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_20 = approx_exp2((-_cvt_f32_41.x) * 1.442695f);
        float e1_20 = approx_exp2((-_cvt_f32_41.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(r0_20), "f"(_fma_111), "f"(r0_20));
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(r1_20), "f"(_fma_113), "f"(r1_20));
        float2 _f32x2_mul_a_114 = mul_f32x2_noftz(_cvt_f32_41, make_float2(_fma_110, _fma_112));
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_cvt_f32_42));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y)));
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_21 = approx_exp2((-_cvt_f32_43.x) * 1.442695f);
        float e1_21 = approx_exp2((-_cvt_f32_43.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(r0_21), "f"(_fma_116), "f"(r0_21));
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(r1_21), "f"(_fma_118), "f"(r1_21));
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(_cvt_f32_43, make_float2(_fma_115, _fma_117));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&_cvt_f32_44));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y)));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_22 = approx_exp2((-_cvt_f32_45.x) * 1.442695f);
        float e1_22 = approx_exp2((-_cvt_f32_45.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(r0_22), "f"(_fma_121), "f"(r0_22));
        float _fma_123;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(r1_22), "f"(_fma_123), "f"(r1_22));
        float2 _f32x2_mul_a_124 = mul_f32x2_noftz(_cvt_f32_45, make_float2(_fma_120, _fma_122));
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&_cvt_f32_46));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y)));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_23 = approx_exp2((-_cvt_f32_47.x) * 1.442695f);
        float e1_23 = approx_exp2((-_cvt_f32_47.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(r0_23), "f"(_fma_126), "f"(r0_23));
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(r1_23), "f"(_fma_128), "f"(r1_23));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(_cvt_f32_47, make_float2(_fma_125, _fma_127));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_cvt_f32_48));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y)));
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
        float e0_24 = approx_exp2((-_cvt_f32_49.x) * 1.442695f);
        float e1_24 = approx_exp2((-_cvt_f32_49.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(r0_24), "f"(_fma_131), "f"(r0_24));
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(r1_24), "f"(_fma_133), "f"(r1_24));
        float2 _f32x2_mul_a_134 = mul_f32x2_noftz(_cvt_f32_49, make_float2(_fma_130, _fma_132));
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&_cvt_f32_50));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y)));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float e0_25 = approx_exp2((-_cvt_f32_51.x) * 1.442695f);
        float e1_25 = approx_exp2((-_cvt_f32_51.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(r0_25), "f"(_fma_136), "f"(r0_25));
        float _fma_138;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(r1_25), "f"(_fma_138), "f"(r1_25));
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(_cvt_f32_51, make_float2(_fma_135, _fma_137));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_cvt_f32_52));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y)));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float e0_26 = approx_exp2((-_cvt_f32_53.x) * 1.442695f);
        float e1_26 = approx_exp2((-_cvt_f32_53.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(r0_26), "f"(_fma_141), "f"(r0_26));
        float _fma_143;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(r1_26), "f"(_fma_143), "f"(r1_26));
        float2 _f32x2_mul_a_144 = mul_f32x2_noftz(_cvt_f32_53, make_float2(_fma_140, _fma_142));
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&_cvt_f32_54));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y)));
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float e0_27 = approx_exp2((-_cvt_f32_55.x) * 1.442695f);
        float e1_27 = approx_exp2((-_cvt_f32_55.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(r0_27), "f"(_fma_146), "f"(r0_27));
        float _fma_148;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(r1_27), "f"(_fma_148), "f"(r1_27));
        float2 _f32x2_mul_a_149 = mul_f32x2_noftz(_cvt_f32_55, make_float2(_fma_145, _fma_147));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&_cvt_f32_56));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y)));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float e0_28 = approx_exp2((-_cvt_f32_57.x) * 1.442695f);
        float e1_28 = approx_exp2((-_cvt_f32_57.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(r0_28), "f"(_fma_151), "f"(r0_28));
        float _fma_153;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(r1_28), "f"(_fma_153), "f"(r1_28));
        float2 _f32x2_mul_a_154 = mul_f32x2_noftz(_cvt_f32_57, make_float2(_fma_150, _fma_152));
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_cvt_f32_58));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y)));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float e0_29 = approx_exp2((-_cvt_f32_59.x) * 1.442695f);
        float e1_29 = approx_exp2((-_cvt_f32_59.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(r0_29), "f"(_fma_156), "f"(r0_29));
        float _fma_158;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(r1_29), "f"(_fma_158), "f"(r1_29));
        float2 _f32x2_mul_a_159 = mul_f32x2_noftz(_cvt_f32_59, make_float2(_fma_155, _fma_157));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&_cvt_f32_60));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y)));
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float e0_30 = approx_exp2((-_cvt_f32_61.x) * 1.442695f);
        float e1_30 = approx_exp2((-_cvt_f32_61.y) * 1.442695f);
        float d0_30 = 1.0f + e0_30;
        float d1_30 = 1.0f + e1_30;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(r0_30), "f"(_fma_161), "f"(r0_30));
        float _fma_163;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(r1_30), "f"(_fma_163), "f"(r1_30));
        float2 _f32x2_mul_a_164 = mul_f32x2_noftz(_cvt_f32_61, make_float2(_fma_160, _fma_162));
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_cvt_f32_62));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y)));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float e0_31 = approx_exp2((-_cvt_f32_63.x) * 1.442695f);
        float e1_31 = approx_exp2((-_cvt_f32_63.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(r0_31), "f"(_fma_166), "f"(r0_31));
        float _fma_168;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(r1_31), "f"(_fma_168), "f"(r1_31));
        float2 _f32x2_mul_a_169 = mul_f32x2_noftz(_cvt_f32_63, make_float2(_fma_165, _fma_167));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&_cvt_f32_64));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y)));
    } else {
        int band = 0;
        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_32 = approx_exp2((-_cvt_f32_65.x) * 1.442695f);
        float e1_32 = approx_exp2((-_cvt_f32_65.y) * 1.442695f);
        float d0_32 = 1.0f + e0_32;
        float d1_32 = 1.0f + e1_32;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_32), "f"(r0_32), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_32), "f"(_fma_170), "f"(r0_32));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(-d1_32), "f"(r1_32), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_32), "f"(_fma_171), "f"(r1_32));
        float t0_32 = _fma_0;
        float t1_32 = _fma_1;
        band = band | ((fminf(t0_32, t1_32) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_172 = mul_f32x2_noftz(_cvt_f32_65, make_float2(max_noftz(t0_32, 0.0f), max_noftz(t1_32, 0.0f)));
        float2 _mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_cvt_f32_66));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y)));
        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_33 = approx_exp2((-_cvt_f32_67.x) * 1.442695f);
        float e1_33 = approx_exp2((-_cvt_f32_67.y) * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_173;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_33), "f"(_fma_173), "f"(r0_33));
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_33), "f"(_fma_174), "f"(r1_33));
        float t0_33 = _fma_2;
        float t1_33 = _fma_3;
        band = band | ((fminf(t0_33, t1_33) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_175 = mul_f32x2_noftz(_cvt_f32_67, make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f)));
        float2 _mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&_cvt_f32_68));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y)));
        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_34 = approx_exp2((-_cvt_f32_69.x) * 1.442695f);
        float e1_34 = approx_exp2((-_cvt_f32_69.y) * 1.442695f);
        float d0_34 = 1.0f + e0_34;
        float d1_34 = 1.0f + e1_34;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float _fma_176;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(-d0_34), "f"(r0_34), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_34), "f"(_fma_176), "f"(r0_34));
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d1_34), "f"(r1_34), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r1_34), "f"(_fma_177), "f"(r1_34));
        float t0_34 = _fma_4;
        float t1_34 = _fma_5;
        band = band | ((fminf(t0_34, t1_34) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_178 = mul_f32x2_noftz(_cvt_f32_69, make_float2(max_noftz(t0_34, 0.0f), max_noftz(t1_34, 0.0f)));
        float2 _mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&_cvt_f32_70));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_34.x, _mul_34.y)));
        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_35 = approx_exp2((-_cvt_f32_71.x) * 1.442695f);
        float e1_35 = approx_exp2((-_cvt_f32_71.y) * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_179;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r0_35), "f"(_fma_179), "f"(r0_35));
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1_35), "f"(_fma_180), "f"(r1_35));
        float t0_35 = _fma_6;
        float t1_35 = _fma_7;
        band = band | ((fminf(t0_35, t1_35) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_181 = mul_f32x2_noftz(_cvt_f32_71, make_float2(max_noftz(t0_35, 0.0f), max_noftz(t1_35, 0.0f)));
        float2 _mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&_cvt_f32_72));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_35.x, _mul_35.y)));
        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_36 = approx_exp2((-_cvt_f32_73.x) * 1.442695f);
        float e1_36 = approx_exp2((-_cvt_f32_73.y) * 1.442695f);
        float d0_36 = 1.0f + e0_36;
        float d1_36 = 1.0f + e1_36;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float _fma_182;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(-d0_36), "f"(r0_36), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r0_36), "f"(_fma_182), "f"(r0_36));
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(-d1_36), "f"(r1_36), "f"(1.0f));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r1_36), "f"(_fma_183), "f"(r1_36));
        float t0_36 = _fma_8;
        float t1_36 = _fma_9;
        band = band | ((fminf(t0_36, t1_36) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_184 = mul_f32x2_noftz(_cvt_f32_73, make_float2(max_noftz(t0_36, 0.0f), max_noftz(t1_36, 0.0f)));
        float2 _mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&_cvt_f32_74));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_36.x, _mul_36.y)));
        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_37 = approx_exp2((-_cvt_f32_75.x) * 1.442695f);
        float e1_37 = approx_exp2((-_cvt_f32_75.y) * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_185;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float _fma_10_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10_1) : "f"(r0_37), "f"(_fma_185), "f"(r0_37));
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float _fma_11_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11_1) : "f"(r1_37), "f"(_fma_186), "f"(r1_37));
        float t0_37 = _fma_10_1;
        float t1_37 = _fma_11_1;
        band = band | ((fminf(t0_37, t1_37) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_187 = mul_f32x2_noftz(_cvt_f32_75, make_float2(max_noftz(t0_37, 0.0f), max_noftz(t1_37, 0.0f)));
        float2 _mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&_cvt_f32_76));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_37.x, _mul_37.y)));
        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_38 = approx_exp2((-_cvt_f32_77.x) * 1.442695f);
        float e1_38 = approx_exp2((-_cvt_f32_77.y) * 1.442695f);
        float d0_38 = 1.0f + e0_38;
        float d1_38 = 1.0f + e1_38;
        float r0_38 = approx_rcp(d0_38);
        float r1_38 = approx_rcp(d1_38);
        float _fma_188;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(-d0_38), "f"(r0_38), "f"(1.0f));
        float _fma_12_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12_1) : "f"(r0_38), "f"(_fma_188), "f"(r0_38));
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d1_38), "f"(r1_38), "f"(1.0f));
        float _fma_13_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13_1) : "f"(r1_38), "f"(_fma_189), "f"(r1_38));
        float t0_38 = _fma_12_1;
        float t1_38 = _fma_13_1;
        band = band | ((fminf(t0_38, t1_38) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_190 = mul_f32x2_noftz(_cvt_f32_77, make_float2(max_noftz(t0_38, 0.0f), max_noftz(t1_38, 0.0f)));
        float2 _mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&_cvt_f32_78));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_38.x, _mul_38.y)));
        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_39 = approx_exp2((-_cvt_f32_79.x) * 1.442695f);
        float e1_39 = approx_exp2((-_cvt_f32_79.y) * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_191;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0_39), "f"(_fma_191), "f"(r0_39));
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float _fma_15_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15_1) : "f"(r1_39), "f"(_fma_192), "f"(r1_39));
        float t0_39 = _fma_14;
        float t1_39 = _fma_15_1;
        band = band | ((fminf(t0_39, t1_39) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_193 = mul_f32x2_noftz(_cvt_f32_79, make_float2(max_noftz(t0_39, 0.0f), max_noftz(t1_39, 0.0f)));
        float2 _mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&_cvt_f32_80));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_39.x, _mul_39.y)));
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
                    float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                    float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                    float e0_40 = approx_exp2((-_cvt_f32_81.x) * 1.442695f);
                    float e1_40 = approx_exp2((-_cvt_f32_81.y) * 1.442695f);
                    float d0_40 = 1.0f + e0_40;
                    float d1_40 = 1.0f + e1_40;
                    float r0_40 = approx_rcp(d0_40);
                    float r1_40 = approx_rcp(d1_40);
                    float _fma_194;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(-d0_40), "f"(r0_40), "f"(1.0f));
                    float _fma_16_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_40), "f"(_fma_194), "f"(r0_40));
                    float _fma_195;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d1_40), "f"(r1_40), "f"(1.0f));
                    float _fma_17_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17_1) : "f"(r1_40), "f"(_fma_195), "f"(r1_40));
                    float t0_40 = _fma_16_1;
                    float t1_40 = _fma_17_1;
                    float d4_40a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_40a) : "f"(d0_40), "f"(0.25f));
                    float r4_40a = approx_rcp(d4_40a);
                    float _fma_197;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(-d4_40a), "f"(r4_40a), "f"(1.0f));
                    float _fma_196;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(r4_40a), "f"(_fma_197), "f"(r4_40a));
                    float big_40a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_40a) : "f"(mul_f32_rn_noftz(_fma_196, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_198;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(big_40a), "f"(8.388608e+06f));
                    float whole_40a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_40a) : "f"(_add_198), "f"(8.388608e+06f));
                    float above_40a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_40a) : "f"(-d4_40a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_40a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_200;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_200) : "f"(big_40a), "f"(whole_40a));
                    float _abs_199;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_199) : "f"(_sub_200));
                    float _add_201;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(big_40a), "f"(0.5f));
                    float _sub_202;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_202) : "f"(big_40a), "f"(0.5f));
                    float rounded_40a = ((_abs_199 == 0.5f) ? ((above_40a > 0.0f) ? _add_201 : _sub_202) : whole_40a);
                    float u0_40 = ((d0_40 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_40a, 7.888609e-31f), 1.7763568e-15f) : t0_40);
                    float d4_40b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_40b) : "f"(d1_40), "f"(0.25f));
                    float r4_40b = approx_rcp(d4_40b);
                    float _fma_204;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(-d4_40b), "f"(r4_40b), "f"(1.0f));
                    float _fma_203;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(r4_40b), "f"(_fma_204), "f"(r4_40b));
                    float big_40b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_40b) : "f"(mul_f32_rn_noftz(_fma_203, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(big_40b), "f"(8.388608e+06f));
                    float whole_40b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_40b) : "f"(_add_205), "f"(8.388608e+06f));
                    float above_40b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_40b) : "f"(-d4_40b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_40b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_207;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_207) : "f"(big_40b), "f"(whole_40b));
                    float _abs_206;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_206) : "f"(_sub_207));
                    float _add_208;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(big_40b), "f"(0.5f));
                    float _sub_209;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_209) : "f"(big_40b), "f"(0.5f));
                    float rounded_40b = ((_abs_206 == 0.5f) ? ((above_40b > 0.0f) ? _add_208 : _sub_209) : whole_40b);
                    float u1_40 = ((d1_40 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_40b, 7.888609e-31f), 1.7763568e-15f) : t1_40);
                    float2 _f32x2_mul_a_210 = mul_f32x2_noftz(_cvt_f32_81, make_float2(max_noftz(u0_40, 0.0f), max_noftz(u1_40, 0.0f)));
                    float2 _mul_40;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&_cvt_f32_82));
                    unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(_mul_40.x, _mul_40.y)));
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
        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0_41 = approx_exp2((-_cvt_f32_83.x) * 1.442695f);
        float e1_41 = approx_exp2((-_cvt_f32_83.y) * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_211;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float _fma_18_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18_1) : "f"(r0_41), "f"(_fma_211), "f"(r0_41));
        float _fma_212;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r1_41), "f"(_fma_212), "f"(r1_41));
        float t0_41 = _fma_18_1;
        float t1_41 = _fma_19;
        band_0 = band_0 | ((fminf(t0_41, t1_41) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_213 = mul_f32x2_noftz(_cvt_f32_83, make_float2(max_noftz(t0_41, 0.0f), max_noftz(t1_41, 0.0f)));
        float2 _mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&_cvt_f32_84));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_41.x, _mul_41.y)));
        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_42 = approx_exp2((-_cvt_f32_85.x) * 1.442695f);
        float e1_42 = approx_exp2((-_cvt_f32_85.y) * 1.442695f);
        float d0_42 = 1.0f + e0_42;
        float d1_42 = 1.0f + e1_42;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float _fma_214;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(-d0_42), "f"(r0_42), "f"(1.0f));
        float _fma_20_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20_1) : "f"(r0_42), "f"(_fma_214), "f"(r0_42));
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d1_42), "f"(r1_42), "f"(1.0f));
        float _fma_21_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21_1) : "f"(r1_42), "f"(_fma_215), "f"(r1_42));
        float t0_42 = _fma_20_1;
        float t1_42 = _fma_21_1;
        band_0 = band_0 | ((fminf(t0_42, t1_42) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_216 = mul_f32x2_noftz(_cvt_f32_85, make_float2(max_noftz(t0_42, 0.0f), max_noftz(t1_42, 0.0f)));
        float2 _mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&_cvt_f32_86));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_42.x, _mul_42.y)));
        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_43 = approx_exp2((-_cvt_f32_87.x) * 1.442695f);
        float e1_43 = approx_exp2((-_cvt_f32_87.y) * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_217;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float _fma_22_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22_1) : "f"(r0_43), "f"(_fma_217), "f"(r0_43));
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float _fma_23_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23_1) : "f"(r1_43), "f"(_fma_218), "f"(r1_43));
        float t0_43 = _fma_22_1;
        float t1_43 = _fma_23_1;
        band_0 = band_0 | ((fminf(t0_43, t1_43) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_219 = mul_f32x2_noftz(_cvt_f32_87, make_float2(max_noftz(t0_43, 0.0f), max_noftz(t1_43, 0.0f)));
        float2 _mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&_cvt_f32_88));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_43.x, _mul_43.y)));
        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_44 = approx_exp2((-_cvt_f32_89.x) * 1.442695f);
        float e1_44 = approx_exp2((-_cvt_f32_89.y) * 1.442695f);
        float d0_44 = 1.0f + e0_44;
        float d1_44 = 1.0f + e1_44;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float _fma_220;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(-d0_44), "f"(r0_44), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_44), "f"(_fma_220), "f"(r0_44));
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(-d1_44), "f"(r1_44), "f"(1.0f));
        float _fma_25_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25_1) : "f"(r1_44), "f"(_fma_221), "f"(r1_44));
        float t0_44 = _fma_24;
        float t1_44 = _fma_25_1;
        band_0 = band_0 | ((fminf(t0_44, t1_44) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_222 = mul_f32x2_noftz(_cvt_f32_89, make_float2(max_noftz(t0_44, 0.0f), max_noftz(t1_44, 0.0f)));
        float2 _mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&_cvt_f32_90));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_44.x, _mul_44.y)));
        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_45 = approx_exp2((-_cvt_f32_91.x) * 1.442695f);
        float e1_45 = approx_exp2((-_cvt_f32_91.y) * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_223;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float _fma_26_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26_1) : "f"(r0_45), "f"(_fma_223), "f"(r0_45));
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float _fma_27_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27_1) : "f"(r1_45), "f"(_fma_224), "f"(r1_45));
        float t0_45 = _fma_26_1;
        float t1_45 = _fma_27_1;
        band_0 = band_0 | ((fminf(t0_45, t1_45) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_225 = mul_f32x2_noftz(_cvt_f32_91, make_float2(max_noftz(t0_45, 0.0f), max_noftz(t1_45, 0.0f)));
        float2 _mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&_cvt_f32_92));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_45.x, _mul_45.y)));
        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_46 = approx_exp2((-_cvt_f32_93.x) * 1.442695f);
        float e1_46 = approx_exp2((-_cvt_f32_93.y) * 1.442695f);
        float d0_46 = 1.0f + e0_46;
        float d1_46 = 1.0f + e1_46;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float _fma_226;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(-d0_46), "f"(r0_46), "f"(1.0f));
        float _fma_28_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28_1) : "f"(r0_46), "f"(_fma_226), "f"(r0_46));
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(-d1_46), "f"(r1_46), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r1_46), "f"(_fma_227), "f"(r1_46));
        float t0_46 = _fma_28_1;
        float t1_46 = _fma_29;
        band_0 = band_0 | ((fminf(t0_46, t1_46) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_228 = mul_f32x2_noftz(_cvt_f32_93, make_float2(max_noftz(t0_46, 0.0f), max_noftz(t1_46, 0.0f)));
        float2 _mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&_cvt_f32_94));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_46.x, _mul_46.y)));
        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_47 = approx_exp2((-_cvt_f32_95.x) * 1.442695f);
        float e1_47 = approx_exp2((-_cvt_f32_95.y) * 1.442695f);
        float d0_47 = 1.0f + e0_47;
        float d1_47 = 1.0f + e1_47;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float _fma_229;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
        float _fma_30_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30_1) : "f"(r0_47), "f"(_fma_229), "f"(r0_47));
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
        float _fma_31_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31_1) : "f"(r1_47), "f"(_fma_230), "f"(r1_47));
        float t0_47 = _fma_30_1;
        float t1_47 = _fma_31_1;
        band_0 = band_0 | ((fminf(t0_47, t1_47) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_231 = mul_f32x2_noftz(_cvt_f32_95, make_float2(max_noftz(t0_47, 0.0f), max_noftz(t1_47, 0.0f)));
        float2 _mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&_cvt_f32_96));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_47.x, _mul_47.y)));
        float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_48 = approx_exp2((-_cvt_f32_97.x) * 1.442695f);
        float e1_48 = approx_exp2((-_cvt_f32_97.y) * 1.442695f);
        float d0_48 = 1.0f + e0_48;
        float d1_48 = 1.0f + e1_48;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float _fma_232;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(-d0_48), "f"(r0_48), "f"(1.0f));
        float _fma_32_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32_1) : "f"(r0_48), "f"(_fma_232), "f"(r0_48));
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d1_48), "f"(r1_48), "f"(1.0f));
        float _fma_33_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33_1) : "f"(r1_48), "f"(_fma_233), "f"(r1_48));
        float t0_48 = _fma_32_1;
        float t1_48 = _fma_33_1;
        band_0 = band_0 | ((fminf(t0_48, t1_48) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_234 = mul_f32x2_noftz(_cvt_f32_97, make_float2(max_noftz(t0_48, 0.0f), max_noftz(t1_48, 0.0f)));
        float2 _mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&_cvt_f32_98));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_48.x, _mul_48.y)));
        if (band_0 != 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 8; i_1++) {
                if ((band_0 >> i_1 & 1) != 0) {
                    unsigned int _vec_load_10[1];
                    {
                        _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_0 + 2 * i_1) + 0);
                    }
                    unsigned int _vec_load_11[1];
                    {
                        _vec_load_11[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_0 + K + 2 * i_1) + 0);
                    }
                    float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
                    float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
                    float e0_49 = approx_exp2((-_cvt_f32_99.x) * 1.442695f);
                    float e1_49 = approx_exp2((-_cvt_f32_99.y) * 1.442695f);
                    float d0_49 = 1.0f + e0_49;
                    float d1_49 = 1.0f + e1_49;
                    float r0_49 = approx_rcp(d0_49);
                    float r1_49 = approx_rcp(d1_49);
                    float _fma_235;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
                    float _fma_34;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_49), "f"(_fma_235), "f"(r0_49));
                    float _fma_236;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
                    float _fma_35_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35_1) : "f"(r1_49), "f"(_fma_236), "f"(r1_49));
                    float t0_49 = _fma_34;
                    float t1_49 = _fma_35_1;
                    float d4_49a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_49a) : "f"(d0_49), "f"(0.25f));
                    float r4_49a = approx_rcp(d4_49a);
                    float _fma_238;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(-d4_49a), "f"(r4_49a), "f"(1.0f));
                    float _fma_237;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(r4_49a), "f"(_fma_238), "f"(r4_49a));
                    float big_49a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_49a) : "f"(mul_f32_rn_noftz(_fma_237, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_239;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(big_49a), "f"(8.388608e+06f));
                    float whole_49a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_49a) : "f"(_add_239), "f"(8.388608e+06f));
                    float above_49a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_49a) : "f"(-d4_49a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_49a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_241;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_241) : "f"(big_49a), "f"(whole_49a));
                    float _abs_240;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_240) : "f"(_sub_241));
                    float _add_242;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(big_49a), "f"(0.5f));
                    float _sub_243;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_243) : "f"(big_49a), "f"(0.5f));
                    float rounded_49a = ((_abs_240 == 0.5f) ? ((above_49a > 0.0f) ? _add_242 : _sub_243) : whole_49a);
                    float u0_49 = ((d0_49 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_49a, 7.888609e-31f), 1.7763568e-15f) : t0_49);
                    float d4_49b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_49b) : "f"(d1_49), "f"(0.25f));
                    float r4_49b = approx_rcp(d4_49b);
                    float _fma_245;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(-d4_49b), "f"(r4_49b), "f"(1.0f));
                    float _fma_244;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(r4_49b), "f"(_fma_245), "f"(r4_49b));
                    float big_49b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_49b) : "f"(mul_f32_rn_noftz(_fma_244, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_246;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(big_49b), "f"(8.388608e+06f));
                    float whole_49b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_49b) : "f"(_add_246), "f"(8.388608e+06f));
                    float above_49b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_49b) : "f"(-d4_49b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_49b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_248;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(big_49b), "f"(whole_49b));
                    float _abs_247;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_247) : "f"(_sub_248));
                    float _add_249;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(big_49b), "f"(0.5f));
                    float _sub_250;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_250) : "f"(big_49b), "f"(0.5f));
                    float rounded_49b = ((_abs_247 == 0.5f) ? ((above_49b > 0.0f) ? _add_249 : _sub_250) : whole_49b);
                    float u1_49 = ((d1_49 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_49b, 7.888609e-31f), 1.7763568e-15f) : t1_49);
                    float2 _f32x2_mul_a_251 = mul_f32x2_noftz(_cvt_f32_99, make_float2(max_noftz(u0_49, 0.0f), max_noftz(u1_49, 0.0f)));
                    float2 _mul_49;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_251), "l"(*(const unsigned long long*)&_cvt_f32_100));
                    unsigned int pair_1 = __as_u32(__float22bfloat162_rn(make_float2(_mul_49.x, _mul_49.y)));
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
        float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_50 = approx_exp2((-_cvt_f32_101.x) * 1.442695f);
        float e1_50 = approx_exp2((-_cvt_f32_101.y) * 1.442695f);
        float d0_50 = 1.0f + e0_50;
        float d1_50 = 1.0f + e1_50;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float _fma_252;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(-d0_50), "f"(r0_50), "f"(1.0f));
        float _fma_36_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36_1) : "f"(r0_50), "f"(_fma_252), "f"(r0_50));
        float _fma_253;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(-d1_50), "f"(r1_50), "f"(1.0f));
        float _fma_37_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37_1) : "f"(r1_50), "f"(_fma_253), "f"(r1_50));
        float t0_50 = _fma_36_1;
        float t1_50 = _fma_37_1;
        band_1 = band_1 | ((fminf(t0_50, t1_50) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_254 = mul_f32x2_noftz(_cvt_f32_101, make_float2(max_noftz(t0_50, 0.0f), max_noftz(t1_50, 0.0f)));
        float2 _mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_254), "l"(*(const unsigned long long*)&_cvt_f32_102));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_50.x, _mul_50.y)));
        float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_51 = approx_exp2((-_cvt_f32_103.x) * 1.442695f);
        float e1_51 = approx_exp2((-_cvt_f32_103.y) * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_255;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float _fma_38_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38_1) : "f"(r0_51), "f"(_fma_255), "f"(r0_51));
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r1_51), "f"(_fma_256), "f"(r1_51));
        float t0_51 = _fma_38_1;
        float t1_51 = _fma_39;
        band_1 = band_1 | ((fminf(t0_51, t1_51) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_257 = mul_f32x2_noftz(_cvt_f32_103, make_float2(max_noftz(t0_51, 0.0f), max_noftz(t1_51, 0.0f)));
        float2 _mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_257), "l"(*(const unsigned long long*)&_cvt_f32_104));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_51.x, _mul_51.y)));
        float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_52 = approx_exp2((-_cvt_f32_105.x) * 1.442695f);
        float e1_52 = approx_exp2((-_cvt_f32_105.y) * 1.442695f);
        float d0_52 = 1.0f + e0_52;
        float d1_52 = 1.0f + e1_52;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float _fma_258;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(-d0_52), "f"(r0_52), "f"(1.0f));
        float _fma_40_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40_1) : "f"(r0_52), "f"(_fma_258), "f"(r0_52));
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d1_52), "f"(r1_52), "f"(1.0f));
        float _fma_41_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41_1) : "f"(r1_52), "f"(_fma_259), "f"(r1_52));
        float t0_52 = _fma_40_1;
        float t1_52 = _fma_41_1;
        band_1 = band_1 | ((fminf(t0_52, t1_52) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_260 = mul_f32x2_noftz(_cvt_f32_105, make_float2(max_noftz(t0_52, 0.0f), max_noftz(t1_52, 0.0f)));
        float2 _mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_260), "l"(*(const unsigned long long*)&_cvt_f32_106));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_52.x, _mul_52.y)));
        float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_53 = approx_exp2((-_cvt_f32_107.x) * 1.442695f);
        float e1_53 = approx_exp2((-_cvt_f32_107.y) * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_261;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float _fma_42_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42_1) : "f"(r0_53), "f"(_fma_261), "f"(r0_53));
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float _fma_43_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43_1) : "f"(r1_53), "f"(_fma_262), "f"(r1_53));
        float t0_53 = _fma_42_1;
        float t1_53 = _fma_43_1;
        band_1 = band_1 | ((fminf(t0_53, t1_53) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_263 = mul_f32x2_noftz(_cvt_f32_107, make_float2(max_noftz(t0_53, 0.0f), max_noftz(t1_53, 0.0f)));
        float2 _mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_263), "l"(*(const unsigned long long*)&_cvt_f32_108));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_53.x, _mul_53.y)));
        float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_54 = approx_exp2((-_cvt_f32_109.x) * 1.442695f);
        float e1_54 = approx_exp2((-_cvt_f32_109.y) * 1.442695f);
        float d0_54 = 1.0f + e0_54;
        float d1_54 = 1.0f + e1_54;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float _fma_264;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(-d0_54), "f"(r0_54), "f"(1.0f));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(r0_54), "f"(_fma_264), "f"(r0_54));
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(-d1_54), "f"(r1_54), "f"(1.0f));
        float _fma_45_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45_1) : "f"(r1_54), "f"(_fma_265), "f"(r1_54));
        float t0_54 = _fma_44;
        float t1_54 = _fma_45_1;
        band_1 = band_1 | ((fminf(t0_54, t1_54) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_266 = mul_f32x2_noftz(_cvt_f32_109, make_float2(max_noftz(t0_54, 0.0f), max_noftz(t1_54, 0.0f)));
        float2 _mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&_cvt_f32_110));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_54.x, _mul_54.y)));
        float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_55 = approx_exp2((-_cvt_f32_111.x) * 1.442695f);
        float e1_55 = approx_exp2((-_cvt_f32_111.y) * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_267;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float _fma_46_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46_1) : "f"(r0_55), "f"(_fma_267), "f"(r0_55));
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float _fma_47_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47_1) : "f"(r1_55), "f"(_fma_268), "f"(r1_55));
        float t0_55 = _fma_46_1;
        float t1_55 = _fma_47_1;
        band_1 = band_1 | ((fminf(t0_55, t1_55) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_269 = mul_f32x2_noftz(_cvt_f32_111, make_float2(max_noftz(t0_55, 0.0f), max_noftz(t1_55, 0.0f)));
        float2 _mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_269), "l"(*(const unsigned long long*)&_cvt_f32_112));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_55.x, _mul_55.y)));
        float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_56 = approx_exp2((-_cvt_f32_113.x) * 1.442695f);
        float e1_56 = approx_exp2((-_cvt_f32_113.y) * 1.442695f);
        float d0_56 = 1.0f + e0_56;
        float d1_56 = 1.0f + e1_56;
        float r0_56 = approx_rcp(d0_56);
        float r1_56 = approx_rcp(d1_56);
        float _fma_270;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(-d0_56), "f"(r0_56), "f"(1.0f));
        float _fma_48_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48_1) : "f"(r0_56), "f"(_fma_270), "f"(r0_56));
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(-d1_56), "f"(r1_56), "f"(1.0f));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(r1_56), "f"(_fma_271), "f"(r1_56));
        float t0_56 = _fma_48_1;
        float t1_56 = _fma_49;
        band_1 = band_1 | ((fminf(t0_56, t1_56) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_272 = mul_f32x2_noftz(_cvt_f32_113, make_float2(max_noftz(t0_56, 0.0f), max_noftz(t1_56, 0.0f)));
        float2 _mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_272), "l"(*(const unsigned long long*)&_cvt_f32_114));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_56.x, _mul_56.y)));
        float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_57 = approx_exp2((-_cvt_f32_115.x) * 1.442695f);
        float e1_57 = approx_exp2((-_cvt_f32_115.y) * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_273;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float _fma_50_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50_1) : "f"(r0_57), "f"(_fma_273), "f"(r0_57));
        float _fma_274;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float _fma_51_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51_1) : "f"(r1_57), "f"(_fma_274), "f"(r1_57));
        float t0_57 = _fma_50_1;
        float t1_57 = _fma_51_1;
        band_1 = band_1 | ((fminf(t0_57, t1_57) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_275 = mul_f32x2_noftz(_cvt_f32_115, make_float2(max_noftz(t0_57, 0.0f), max_noftz(t1_57, 0.0f)));
        float2 _mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_275), "l"(*(const unsigned long long*)&_cvt_f32_116));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_57.x, _mul_57.y)));
        if (band_1 != 0) {
            #pragma unroll 1
            for (int i_2 = 0; i_2 < 8; i_2++) {
                if ((band_1 >> i_2 & 1) != 0) {
                    unsigned int _vec_load_12[1];
                    {
                        _vec_load_12[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_1 + 2 * i_2) + 0);
                    }
                    unsigned int _vec_load_13[1];
                    {
                        _vec_load_13[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_1 + K + 2 * i_2) + 0);
                    }
                    float2 _cvt_f32_117 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_12[0]));
                    float2 _cvt_f32_118 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_13[0]));
                    float e0_58 = approx_exp2((-_cvt_f32_117.x) * 1.442695f);
                    float e1_58 = approx_exp2((-_cvt_f32_117.y) * 1.442695f);
                    float d0_58 = 1.0f + e0_58;
                    float d1_58 = 1.0f + e1_58;
                    float r0_58 = approx_rcp(d0_58);
                    float r1_58 = approx_rcp(d1_58);
                    float _fma_276;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(-d0_58), "f"(r0_58), "f"(1.0f));
                    float _fma_52_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52_1) : "f"(r0_58), "f"(_fma_276), "f"(r0_58));
                    float _fma_277;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(-d1_58), "f"(r1_58), "f"(1.0f));
                    float _fma_53_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53_1) : "f"(r1_58), "f"(_fma_277), "f"(r1_58));
                    float t0_58 = _fma_52_1;
                    float t1_58 = _fma_53_1;
                    float d4_58a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_58a) : "f"(d0_58), "f"(0.25f));
                    float r4_58a = approx_rcp(d4_58a);
                    float _fma_279;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(-d4_58a), "f"(r4_58a), "f"(1.0f));
                    float _fma_278;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(r4_58a), "f"(_fma_279), "f"(r4_58a));
                    float big_58a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_58a) : "f"(mul_f32_rn_noftz(_fma_278, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_280;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(big_58a), "f"(8.388608e+06f));
                    float whole_58a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_58a) : "f"(_add_280), "f"(8.388608e+06f));
                    float above_58a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_58a) : "f"(-d4_58a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_58a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_282;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(big_58a), "f"(whole_58a));
                    float _abs_281;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_281) : "f"(_sub_282));
                    float _add_283;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(big_58a), "f"(0.5f));
                    float _sub_284;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(big_58a), "f"(0.5f));
                    float rounded_58a = ((_abs_281 == 0.5f) ? ((above_58a > 0.0f) ? _add_283 : _sub_284) : whole_58a);
                    float u0_58 = ((d0_58 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_58a, 7.888609e-31f), 1.7763568e-15f) : t0_58);
                    float d4_58b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_58b) : "f"(d1_58), "f"(0.25f));
                    float r4_58b = approx_rcp(d4_58b);
                    float _fma_286;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(-d4_58b), "f"(r4_58b), "f"(1.0f));
                    float _fma_285;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(r4_58b), "f"(_fma_286), "f"(r4_58b));
                    float big_58b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_58b) : "f"(mul_f32_rn_noftz(_fma_285, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_287;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(big_58b), "f"(8.388608e+06f));
                    float whole_58b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_58b) : "f"(_add_287), "f"(8.388608e+06f));
                    float above_58b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_58b) : "f"(-d4_58b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_58b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_289;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(big_58b), "f"(whole_58b));
                    float _abs_288;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(_sub_289));
                    float _add_290;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(big_58b), "f"(0.5f));
                    float _sub_291;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(big_58b), "f"(0.5f));
                    float rounded_58b = ((_abs_288 == 0.5f) ? ((above_58b > 0.0f) ? _add_290 : _sub_291) : whole_58b);
                    float u1_58 = ((d1_58 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_58b, 7.888609e-31f), 1.7763568e-15f) : t1_58);
                    float2 _f32x2_mul_a_292 = mul_f32x2_noftz(_cvt_f32_117, make_float2(max_noftz(u0_58, 0.0f), max_noftz(u1_58, 0.0f)));
                    float2 _mul_58;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_292), "l"(*(const unsigned long long*)&_cvt_f32_118));
                    unsigned int pair_2 = __as_u32(__float22bfloat162_rn(make_float2(_mul_58.x, _mul_58.y)));
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
        float2 _cvt_f32_119 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
        float2 _cvt_f32_120 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
        float e0_59 = approx_exp2((-_cvt_f32_119.x) * 1.442695f);
        float e1_59 = approx_exp2((-_cvt_f32_119.y) * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_293;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(r0_59), "f"(_fma_293), "f"(r0_59));
        float _fma_294;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float _fma_55_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55_1) : "f"(r1_59), "f"(_fma_294), "f"(r1_59));
        float t0_59 = _fma_54;
        float t1_59 = _fma_55_1;
        band_2 = band_2 | ((fminf(t0_59, t1_59) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_295 = mul_f32x2_noftz(_cvt_f32_119, make_float2(max_noftz(t0_59, 0.0f), max_noftz(t1_59, 0.0f)));
        float2 _mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_295), "l"(*(const unsigned long long*)&_cvt_f32_120));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_59.x, _mul_59.y)));
        float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float e0_60 = approx_exp2((-_cvt_f32_121.x) * 1.442695f);
        float e1_60 = approx_exp2((-_cvt_f32_121.y) * 1.442695f);
        float d0_60 = 1.0f + e0_60;
        float d1_60 = 1.0f + e1_60;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float _fma_296;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(-d0_60), "f"(r0_60), "f"(1.0f));
        float _fma_56_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56_1) : "f"(r0_60), "f"(_fma_296), "f"(r0_60));
        float _fma_297;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(-d1_60), "f"(r1_60), "f"(1.0f));
        float _fma_57_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57_1) : "f"(r1_60), "f"(_fma_297), "f"(r1_60));
        float t0_60 = _fma_56_1;
        float t1_60 = _fma_57_1;
        band_2 = band_2 | ((fminf(t0_60, t1_60) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_298 = mul_f32x2_noftz(_cvt_f32_121, make_float2(max_noftz(t0_60, 0.0f), max_noftz(t1_60, 0.0f)));
        float2 _mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_298), "l"(*(const unsigned long long*)&_cvt_f32_122));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_60.x, _mul_60.y)));
        float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float e0_61 = approx_exp2((-_cvt_f32_123.x) * 1.442695f);
        float e1_61 = approx_exp2((-_cvt_f32_123.y) * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_299;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float _fma_58_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58_1) : "f"(r0_61), "f"(_fma_299), "f"(r0_61));
        float _fma_300;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(r1_61), "f"(_fma_300), "f"(r1_61));
        float t0_61 = _fma_58_1;
        float t1_61 = _fma_59;
        band_2 = band_2 | ((fminf(t0_61, t1_61) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_301 = mul_f32x2_noftz(_cvt_f32_123, make_float2(max_noftz(t0_61, 0.0f), max_noftz(t1_61, 0.0f)));
        float2 _mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_301), "l"(*(const unsigned long long*)&_cvt_f32_124));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_61.x, _mul_61.y)));
        float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float e0_62 = approx_exp2((-_cvt_f32_125.x) * 1.442695f);
        float e1_62 = approx_exp2((-_cvt_f32_125.y) * 1.442695f);
        float d0_62 = 1.0f + e0_62;
        float d1_62 = 1.0f + e1_62;
        float r0_62 = approx_rcp(d0_62);
        float r1_62 = approx_rcp(d1_62);
        float _fma_302;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(-d0_62), "f"(r0_62), "f"(1.0f));
        float _fma_60_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60_1) : "f"(r0_62), "f"(_fma_302), "f"(r0_62));
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d1_62), "f"(r1_62), "f"(1.0f));
        float _fma_61_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61_1) : "f"(r1_62), "f"(_fma_303), "f"(r1_62));
        float t0_62 = _fma_60_1;
        float t1_62 = _fma_61_1;
        band_2 = band_2 | ((fminf(t0_62, t1_62) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_304 = mul_f32x2_noftz(_cvt_f32_125, make_float2(max_noftz(t0_62, 0.0f), max_noftz(t1_62, 0.0f)));
        float2 _mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_304), "l"(*(const unsigned long long*)&_cvt_f32_126));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_62.x, _mul_62.y)));
        float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_128 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float e0_63 = approx_exp2((-_cvt_f32_127.x) * 1.442695f);
        float e1_63 = approx_exp2((-_cvt_f32_127.y) * 1.442695f);
        float d0_63 = 1.0f + e0_63;
        float d1_63 = 1.0f + e1_63;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float _fma_305;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(-d0_63), "f"(r0_63), "f"(1.0f));
        float _fma_62_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62_1) : "f"(r0_63), "f"(_fma_305), "f"(r0_63));
        float _fma_306;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(-d1_63), "f"(r1_63), "f"(1.0f));
        float _fma_63_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63_1) : "f"(r1_63), "f"(_fma_306), "f"(r1_63));
        float t0_63 = _fma_62_1;
        float t1_63 = _fma_63_1;
        band_2 = band_2 | ((fminf(t0_63, t1_63) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_307 = mul_f32x2_noftz(_cvt_f32_127, make_float2(max_noftz(t0_63, 0.0f), max_noftz(t1_63, 0.0f)));
        float2 _mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_307), "l"(*(const unsigned long long*)&_cvt_f32_128));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_63.x, _mul_63.y)));
        float2 _cvt_f32_129 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_130 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float e0_64 = approx_exp2((-_cvt_f32_129.x) * 1.442695f);
        float e1_64 = approx_exp2((-_cvt_f32_129.y) * 1.442695f);
        float d0_64 = 1.0f + e0_64;
        float d1_64 = 1.0f + e1_64;
        float r0_64 = approx_rcp(d0_64);
        float r1_64 = approx_rcp(d1_64);
        float _fma_308;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(-d0_64), "f"(r0_64), "f"(1.0f));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r0_64), "f"(_fma_308), "f"(r0_64));
        float _fma_309;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(-d1_64), "f"(r1_64), "f"(1.0f));
        float _fma_65_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65_1) : "f"(r1_64), "f"(_fma_309), "f"(r1_64));
        float t0_64 = _fma_64;
        float t1_64 = _fma_65_1;
        band_2 = band_2 | ((fminf(t0_64, t1_64) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_310 = mul_f32x2_noftz(_cvt_f32_129, make_float2(max_noftz(t0_64, 0.0f), max_noftz(t1_64, 0.0f)));
        float2 _mul_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_310), "l"(*(const unsigned long long*)&_cvt_f32_130));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_64.x, _mul_64.y)));
        float2 _cvt_f32_131 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_132 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float e0_65 = approx_exp2((-_cvt_f32_131.x) * 1.442695f);
        float e1_65 = approx_exp2((-_cvt_f32_131.y) * 1.442695f);
        float d0_65 = 1.0f + e0_65;
        float d1_65 = 1.0f + e1_65;
        float r0_65 = approx_rcp(d0_65);
        float r1_65 = approx_rcp(d1_65);
        float _fma_311;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(-d0_65), "f"(r0_65), "f"(1.0f));
        float _fma_66_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66_1) : "f"(r0_65), "f"(_fma_311), "f"(r0_65));
        float _fma_312;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(-d1_65), "f"(r1_65), "f"(1.0f));
        float _fma_67_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67_1) : "f"(r1_65), "f"(_fma_312), "f"(r1_65));
        float t0_65 = _fma_66_1;
        float t1_65 = _fma_67_1;
        band_2 = band_2 | ((fminf(t0_65, t1_65) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_313 = mul_f32x2_noftz(_cvt_f32_131, make_float2(max_noftz(t0_65, 0.0f), max_noftz(t1_65, 0.0f)));
        float2 _mul_65;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_313), "l"(*(const unsigned long long*)&_cvt_f32_132));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_65.x, _mul_65.y)));
        float2 _cvt_f32_133 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_134 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float e0_66 = approx_exp2((-_cvt_f32_133.x) * 1.442695f);
        float e1_66 = approx_exp2((-_cvt_f32_133.y) * 1.442695f);
        float d0_66 = 1.0f + e0_66;
        float d1_66 = 1.0f + e1_66;
        float r0_66 = approx_rcp(d0_66);
        float r1_66 = approx_rcp(d1_66);
        float _fma_314;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(-d0_66), "f"(r0_66), "f"(1.0f));
        float _fma_68_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68_1) : "f"(r0_66), "f"(_fma_314), "f"(r0_66));
        float _fma_315;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(-d1_66), "f"(r1_66), "f"(1.0f));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(r1_66), "f"(_fma_315), "f"(r1_66));
        float t0_66 = _fma_68_1;
        float t1_66 = _fma_69;
        band_2 = band_2 | ((fminf(t0_66, t1_66) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_316 = mul_f32x2_noftz(_cvt_f32_133, make_float2(max_noftz(t0_66, 0.0f), max_noftz(t1_66, 0.0f)));
        float2 _mul_66;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_316), "l"(*(const unsigned long long*)&_cvt_f32_134));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_66.x, _mul_66.y)));
        if (band_2 != 0) {
            #pragma unroll 1
            for (int i_3 = 0; i_3 < 8; i_3++) {
                if ((band_2 >> i_3 & 1) != 0) {
                    unsigned int _vec_load_14[1];
                    {
                        _vec_load_14[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_2 + 2 * i_3) + 0);
                    }
                    unsigned int _vec_load_15[1];
                    {
                        _vec_load_15[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_2 + K + 2 * i_3) + 0);
                    }
                    float2 _cvt_f32_135 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_14[0]));
                    float2 _cvt_f32_136 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_15[0]));
                    float e0_67 = approx_exp2((-_cvt_f32_135.x) * 1.442695f);
                    float e1_67 = approx_exp2((-_cvt_f32_135.y) * 1.442695f);
                    float d0_67 = 1.0f + e0_67;
                    float d1_67 = 1.0f + e1_67;
                    float r0_67 = approx_rcp(d0_67);
                    float r1_67 = approx_rcp(d1_67);
                    float _fma_317;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(-d0_67), "f"(r0_67), "f"(1.0f));
                    float _fma_70_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70_1) : "f"(r0_67), "f"(_fma_317), "f"(r0_67));
                    float _fma_318;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(-d1_67), "f"(r1_67), "f"(1.0f));
                    float _fma_71_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71_1) : "f"(r1_67), "f"(_fma_318), "f"(r1_67));
                    float t0_67 = _fma_70_1;
                    float t1_67 = _fma_71_1;
                    float d4_67a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67a) : "f"(d0_67), "f"(0.25f));
                    float r4_67a = approx_rcp(d4_67a);
                    float _fma_320;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(-d4_67a), "f"(r4_67a), "f"(1.0f));
                    float _fma_319;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(r4_67a), "f"(_fma_320), "f"(r4_67a));
                    float big_67a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67a) : "f"(mul_f32_rn_noftz(_fma_319, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_321;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_321) : "f"(big_67a), "f"(8.388608e+06f));
                    float whole_67a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67a) : "f"(_add_321), "f"(8.388608e+06f));
                    float above_67a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67a) : "f"(-d4_67a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_323;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(big_67a), "f"(whole_67a));
                    float _abs_322;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_322) : "f"(_sub_323));
                    float _add_324;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(big_67a), "f"(0.5f));
                    float _sub_325;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(big_67a), "f"(0.5f));
                    float rounded_67a = ((_abs_322 == 0.5f) ? ((above_67a > 0.0f) ? _add_324 : _sub_325) : whole_67a);
                    float u0_67 = ((d0_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67a, 7.888609e-31f), 1.7763568e-15f) : t0_67);
                    float d4_67b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67b) : "f"(d1_67), "f"(0.25f));
                    float r4_67b = approx_rcp(d4_67b);
                    float _fma_327;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(-d4_67b), "f"(r4_67b), "f"(1.0f));
                    float _fma_326;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(r4_67b), "f"(_fma_327), "f"(r4_67b));
                    float big_67b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67b) : "f"(mul_f32_rn_noftz(_fma_326, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_328;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(big_67b), "f"(8.388608e+06f));
                    float whole_67b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67b) : "f"(_add_328), "f"(8.388608e+06f));
                    float above_67b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67b) : "f"(-d4_67b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_330;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(big_67b), "f"(whole_67b));
                    float _abs_329;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_329) : "f"(_sub_330));
                    float _add_331;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(big_67b), "f"(0.5f));
                    float _sub_332;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_332) : "f"(big_67b), "f"(0.5f));
                    float rounded_67b = ((_abs_329 == 0.5f) ? ((above_67b > 0.0f) ? _add_331 : _sub_332) : whole_67b);
                    float u1_67 = ((d1_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67b, 7.888609e-31f), 1.7763568e-15f) : t1_67);
                    float2 _f32x2_mul_a_333 = mul_f32x2_noftz(_cvt_f32_135, make_float2(max_noftz(u0_67, 0.0f), max_noftz(u1_67, 0.0f)));
                    float2 _mul_67;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_333), "l"(*(const unsigned long long*)&_cvt_f32_136));
                    unsigned int pair_3 = __as_u32(__float22bfloat162_rn(make_float2(_mul_67.x, _mul_67.y)));
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
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_351;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_350;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_350) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_351));
    row_words[0] = _mov_b32_pack_u16x2_350;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(packed[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(packed[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_353;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_352;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_352) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_353));
    row_words[1] = _mov_b32_pack_u16x2_352;
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(packed[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(packed[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_355;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_354;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_354) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_355));
    row_words[2] = _mov_b32_pack_u16x2_354;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(packed[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(packed[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_357;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_356;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_356) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_357));
    row_words[3] = _mov_b32_pack_u16x2_356;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
    __nv_bfloat162 amax_3 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_359 = amax_3;
    __nv_bfloat162 _hxb_359 = __as_bf16x2(packed_b[0]);
    unsigned int _hxr_359;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_359) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_359)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_359)));
    __nv_bfloat162 _max_358 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_359);
    amax_3 = _max_358;
    __nv_bfloat162 _hxa_361 = amax_3;
    __nv_bfloat162 _hxb_361 = __as_bf16x2(packed_b[1]);
    unsigned int _hxr_361;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_361) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_361)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_361)));
    __nv_bfloat162 _max_360 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_361);
    amax_3 = _max_360;
    __nv_bfloat162 _hxa_363 = amax_3;
    __nv_bfloat162 _hxb_363 = __as_bf16x2(packed_b[2]);
    unsigned int _hxr_363;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_363) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_363)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_363)));
    __nv_bfloat162 _max_362 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_363);
    amax_3 = _max_362;
    __nv_bfloat162 _hxa_365 = amax_3;
    __nv_bfloat162 _hxb_365 = __as_bf16x2(packed_b[3]);
    unsigned int _hxr_365;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_365) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_365)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_365)));
    __nv_bfloat162 _max_364 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_365);
    amax_3 = _max_364;
    __nv_bfloat162 _hxa_367 = amax_3;
    __nv_bfloat162 _hxb_367 = __as_bf16x2(packed_b[4]);
    unsigned int _hxr_367;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_367) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_367)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_367)));
    __nv_bfloat162 _max_366 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_367);
    amax_3 = _max_366;
    __nv_bfloat162 _hxa_369 = amax_3;
    __nv_bfloat162 _hxb_369 = __as_bf16x2(packed_b[5]);
    unsigned int _hxr_369;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_369) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_369)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_369)));
    __nv_bfloat162 _max_368 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_369);
    amax_3 = _max_368;
    __nv_bfloat162 _hxa_371 = amax_3;
    __nv_bfloat162 _hxb_371 = __as_bf16x2(packed_b[6]);
    unsigned int _hxr_371;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_371) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_371)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_371)));
    __nv_bfloat162 _max_370 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_371);
    amax_3 = _max_370;
    __nv_bfloat162 _hxa_373 = amax_3;
    __nv_bfloat162 _hxb_373 = __as_bf16x2(packed_b[7]);
    unsigned int _hxr_373;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_373) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_373)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_373)));
    __nv_bfloat162 _max_372 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_373);
    amax_3 = _max_372;
    unsigned int mag_4 = __as_u32(amax_3) & 2147450879;
    unsigned int peer_5 = __shfl_xor_sync(0xFFFFFFFF, mag_4, 1);
    __nv_bfloat162 both_6 = __hmax2_nan(__as_bf16x2(mag_4), __as_bf16x2(peer_5));
    unsigned int folded_7 = __as_u32(__hmax2_nan(both_6, __as_bf16x2(__as_u32(both_6) >> 16)));
    unsigned int amax_bits_8 = (folded_7 & 65535) << 16;
    int scale_i32_9 = (int)(amax_bits_8 + 2031616 >> 23) - 8;
    if (scale_i32_9 < 0) {
        scale_i32_9 = 0;
    }
    if ((amax_bits_8 & 2139095040) == 2139095040) {
        scale_i32_9 = 255;
    }
    unsigned int scale_10 = (unsigned int)scale_i32_9;
    unsigned int inv_11 = 254 - scale_10 << 7;
    if (scale_10 == 255) {
        inv_11 = 32704;
    }
    unsigned int inv2_12 = inv_11 | inv_11 << 16;
    int scale_col_13 = bx * 4 + blk;
    int row_scale_index_14 = ((grow_b >> 7) * (K >> 7) + (scale_col_13 >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col_13 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_14) + (0)) = (unsigned char)(scale_10);
    }
    unsigned int row_words_15[4];
    uint32_t _bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(packed_b[0]), "r"(inv2_12));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(packed_b[1]), "r"(inv2_12));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_375;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_375) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_375) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_374;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_374) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_375));
    row_words_15[0] = _mov_b32_pack_u16x2_374;
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(packed_b[2]), "r"(inv2_12));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(packed_b[3]), "r"(inv2_12));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_377;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_377) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_377) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_376;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_376) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_377));
    row_words_15[1] = _mov_b32_pack_u16x2_376;
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(packed_b[4]), "r"(inv2_12));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(packed_b[5]), "r"(inv2_12));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_379;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_379) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_379) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_378;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_378) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_379));
    row_words_15[2] = _mov_b32_pack_u16x2_378;
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(packed_b[6]), "r"(inv2_12));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(packed_b[7]), "r"(inv2_12));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_381;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_381) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_381) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_380;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_380) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_381));
    row_words_15[3] = _mov_b32_pack_u16x2_380;
    reinterpret_cast<int4*>(row_output + (grow_b * K + col))[0] = reinterpret_cast<int4*>(row_words_15 + 0)[0];
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
    unsigned int amax_bits_21 = (folded_20 & 65535) << 16;
    int scale_i32_22 = (int)(amax_bits_21 + 2031616 >> 23) - 8;
    if (scale_i32_22 < 0) {
        scale_i32_22 = 0;
    }
    if ((amax_bits_21 & 2139095040) == 2139095040) {
        scale_i32_22 = 255;
    }
    unsigned int scale_23 = (unsigned int)scale_i32_22;
    unsigned int inv_24 = 254 - scale_23 << 7;
    if (scale_23 == 255) {
        inv_24 = 32704;
    }
    unsigned int inv2_25 = inv_24 | inv_24 << 16;
    int scale_col_26 = bx * 4 + blk;
    int row_scale_index_27 = ((grow_c >> 7) * (K >> 7) + (scale_col_26 >> 2)) * 512 + (grow_c & 31) * 16 + (grow_c >> 5 & 3) * 4 + (scale_col_26 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_27) + (0)) = (unsigned char)(scale_23);
    }
    unsigned int row_words_28[4];
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(packed_c[0]), "r"(inv2_25));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(packed_c[1]), "r"(inv2_25));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_399;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_398;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_398) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_399));
    row_words_28[0] = _mov_b32_pack_u16x2_398;
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(packed_c[2]), "r"(inv2_25));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(packed_c[3]), "r"(inv2_25));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_401;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_401) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_401) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_400;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_400) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_401));
    row_words_28[1] = _mov_b32_pack_u16x2_400;
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(packed_c[4]), "r"(inv2_25));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(packed_c[5]), "r"(inv2_25));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_403;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_402;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_402) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_403));
    row_words_28[2] = _mov_b32_pack_u16x2_402;
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(packed_c[6]), "r"(inv2_25));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(packed_c[7]), "r"(inv2_25));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_405;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_404;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_404) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_405));
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
    unsigned int amax_bits_34 = (folded_33 & 65535) << 16;
    int scale_i32_35 = (int)(amax_bits_34 + 2031616 >> 23) - 8;
    if (scale_i32_35 < 0) {
        scale_i32_35 = 0;
    }
    if ((amax_bits_34 & 2139095040) == 2139095040) {
        scale_i32_35 = 255;
    }
    unsigned int scale_36 = (unsigned int)scale_i32_35;
    unsigned int inv_37 = 254 - scale_36 << 7;
    if (scale_36 == 255) {
        inv_37 = 32704;
    }
    unsigned int inv2_38 = inv_37 | inv_37 << 16;
    int scale_col_39 = bx * 4 + blk;
    int row_scale_index_40 = ((grow_d >> 7) * (K >> 7) + (scale_col_39 >> 2)) * 512 + (grow_d & 31) * 16 + (grow_d >> 5 & 3) * 4 + (scale_col_39 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_40) + (0)) = (unsigned char)(scale_36);
    }
    unsigned int row_words_41[4];
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(packed_d[0]), "r"(inv2_38));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(packed_d[1]), "r"(inv2_38));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_423;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_422;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_422) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_423));
    row_words_41[0] = _mov_b32_pack_u16x2_422;
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(packed_d[2]), "r"(inv2_38));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(packed_d[3]), "r"(inv2_38));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_425;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_424;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_424) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_425));
    row_words_41[1] = _mov_b32_pack_u16x2_424;
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(packed_d[4]), "r"(inv2_38));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(packed_d[5]), "r"(inv2_38));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_427;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_426;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_426) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_427));
    row_words_41[2] = _mov_b32_pack_u16x2_426;
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(packed_d[6]), "r"(inv2_38));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(packed_d[7]), "r"(inv2_38));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_429;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_428;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_428) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_429));
    row_words_41[3] = _mov_b32_pack_u16x2_428;
    reinterpret_cast<int4*>(row_output + (grow_d * K + col))[0] = reinterpret_cast<int4*>(row_words_41 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 32 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 4 & 6 | srow & 1);
    unsigned int cols[32];
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0 + 6144)
        : "memory");
    unsigned int strip_chunk1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src1 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src1 + 6144)
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
    unsigned int amax_bits_42 = (col_max_bits & 65535) << 16;
    int scale_i32_43 = (int)(amax_bits_42 + 2031616 >> 23) - 8;
    if (scale_i32_43 < 0) {
        scale_i32_43 = 0;
    }
    if ((amax_bits_42 & 2139095040) == 2139095040) {
        scale_i32_43 = 255;
    }
    unsigned int scale_44 = (unsigned int)scale_i32_43;
    unsigned int inv_45 = 254 - scale_44 << 7;
    if (scale_44 == 255) {
        inv_45 = 32704;
    }
    unsigned int inv2_46 = inv_45 | inv_45 << 16;
    unsigned int amax_bits_47 = (col_max_bits >> 16 & 65535) << 16;
    int scale_i32_48 = (int)(amax_bits_47 + 2031616 >> 23) - 8;
    if (scale_i32_48 < 0) {
        scale_i32_48 = 0;
    }
    if ((amax_bits_47 & 2139095040) == 2139095040) {
        scale_i32_48 = 255;
    }
    unsigned int scale_49 = (unsigned int)scale_i32_48;
    unsigned int inv_50 = 254 - scale_49 << 7;
    if (scale_49 == 255) {
        inv_50 = 32704;
    }
    unsigned int inv2_51 = inv_50 | inv_50 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
    uint32_t _bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(cols[0]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(cols[1]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_495;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "r"(_bf16x2_mul_33));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_33);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_494;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_494) : "h"(_e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_495));
    words_a[0] = _mov_b32_pack_u16x2_494;
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(cols[16]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(cols[17]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_497;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "r"(_bf16x2_mul_35));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_35);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_496;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_496) : "h"(_e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_497));
    words_b[0] = _mov_b32_pack_u16x2_496;
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(cols[2]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(cols[3]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_499;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "r"(_bf16x2_mul_37));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_37);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_498;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_498) : "h"(_e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_499));
    words_a[1] = _mov_b32_pack_u16x2_498;
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(cols[18]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(cols[19]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_501;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "r"(_bf16x2_mul_39));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_39);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_500;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_500) : "h"(_e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_501));
    words_b[1] = _mov_b32_pack_u16x2_500;
    uint32_t _bf16x2_mul_40;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(cols[4]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(cols[5]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_503;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "r"(_bf16x2_mul_41));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_41);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_502;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_502) : "h"(_e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_503));
    words_a[2] = _mov_b32_pack_u16x2_502;
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(cols[20]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(cols[21]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_505;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "r"(_bf16x2_mul_43));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_43);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_504;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_504) : "h"(_e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_505));
    words_b[2] = _mov_b32_pack_u16x2_504;
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(cols[6]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(cols[7]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_507;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "r"(_bf16x2_mul_45));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_45);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_506;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_506) : "h"(_e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_507));
    words_a[3] = _mov_b32_pack_u16x2_506;
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(cols[22]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(cols[23]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_509;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "r"(_bf16x2_mul_47));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_47);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_508;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_508) : "h"(_e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_509));
    words_b[3] = _mov_b32_pack_u16x2_508;
    uint32_t _bf16x2_mul_48;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(cols[8]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(cols[9]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_511;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "r"(_bf16x2_mul_49));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_49);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_510;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_510) : "h"(_e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_511));
    words_a[4] = _mov_b32_pack_u16x2_510;
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(cols[24]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(cols[25]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_513;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "r"(_bf16x2_mul_51));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_51);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_512;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_512) : "h"(_e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_513));
    words_b[4] = _mov_b32_pack_u16x2_512;
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(cols[10]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(cols[11]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_515;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "r"(_bf16x2_mul_53));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_53);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_514;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_514) : "h"(_e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_515));
    words_a[5] = _mov_b32_pack_u16x2_514;
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(cols[26]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(cols[27]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_517;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "r"(_bf16x2_mul_55));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_55);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_516;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_516) : "h"(_e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_517));
    words_b[5] = _mov_b32_pack_u16x2_516;
    uint32_t _bf16x2_mul_56;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(cols[12]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(cols[13]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_519;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "r"(_bf16x2_mul_57));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_57);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_518;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_518) : "h"(_e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_519));
    words_a[6] = _mov_b32_pack_u16x2_518;
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(cols[28]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(cols[29]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_521;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "r"(_bf16x2_mul_59));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_59);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_520;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_520) : "h"(_e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_521));
    words_b[6] = _mov_b32_pack_u16x2_520;
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(cols[14]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(cols[15]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_523;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "r"(_bf16x2_mul_61));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_61);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_522;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_522) : "h"(_e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_523));
    words_a[7] = _mov_b32_pack_u16x2_522;
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(cols[30]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(cols[31]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_525;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "r"(_bf16x2_mul_63));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_63);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_524;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_524) : "h"(_e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_525));
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
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_44);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_49);
}

} // extern "C"
