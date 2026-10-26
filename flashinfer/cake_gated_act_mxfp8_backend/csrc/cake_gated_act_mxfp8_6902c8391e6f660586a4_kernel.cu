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
kernel_cake_gated_act_mxfp8_6902c8391e6f660586a4(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(_cvt_f32_1, s_0);
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&_cvt_f32_2));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y)));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&_cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(_cvt_f32_3, s_1);
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_cvt_f32_4));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y)));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&_cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(_cvt_f32_5, s_2);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_cvt_f32_6));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y)));
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(_cvt_f32_7, s_3);
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_cvt_f32_8));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y)));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&_cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(_cvt_f32_9, s_4);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_cvt_f32_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y)));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&_cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(_cvt_f32_11, s_5);
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_cvt_f32_12));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y)));
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(_cvt_f32_13, s_6);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&_cvt_f32_14));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y)));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&_cvt_f32_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(_cvt_f32_15, s_7);
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&_cvt_f32_16));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y)));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&_cvt_f32_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_8 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(_cvt_f32_17, s_8);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_cvt_f32_18));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y)));
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(_cvt_f32_19, s_9);
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_cvt_f32_20));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y)));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&_cvt_f32_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_10 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(_cvt_f32_21, s_10);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_cvt_f32_22));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y)));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&_cvt_f32_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(_cvt_f32_23, s_11);
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_cvt_f32_24));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y)));
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_12 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(_cvt_f32_25, s_12);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_cvt_f32_26));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y)));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&_cvt_f32_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(_cvt_f32_27, s_13);
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_cvt_f32_28));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y)));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&_cvt_f32_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_14 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(_cvt_f32_29, s_14);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_cvt_f32_30));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y)));
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_15 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float2 _f32x2_mul_a_41 = mul_f32x2_noftz(_cvt_f32_31, s_15);
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_cvt_f32_32));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y)));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float2 _f32x2_mul_b_42 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&_cvt_f32_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float2 r_16 = make_float2(r0_16, r1_16);
        float2 s_16 = fma_f32x2_rn_noftz(r_16, fma_f32x2_rn_noftz(nd_16, r_16, make_float2(1.0f, 1.0f)), r_16);
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(_cvt_f32_33, s_16);
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_cvt_f32_34));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y)));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_44 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&_cvt_f32_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_17 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(_cvt_f32_35, s_17);
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_cvt_f32_36));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y)));
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float2 r_18 = make_float2(r0_18, r1_18);
        float2 s_18 = fma_f32x2_rn_noftz(r_18, fma_f32x2_rn_noftz(nd_18, r_18, make_float2(1.0f, 1.0f)), r_18);
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(_cvt_f32_37, s_18);
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_cvt_f32_38));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y)));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&_cvt_f32_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_19 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(_cvt_f32_39, s_19);
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&_cvt_f32_40));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y)));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_50 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&_cvt_f32_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float2 r_20 = make_float2(r0_20, r1_20);
        float2 s_20 = fma_f32x2_rn_noftz(r_20, fma_f32x2_rn_noftz(nd_20, r_20, make_float2(1.0f, 1.0f)), r_20);
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(_cvt_f32_41, s_20);
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&_cvt_f32_42));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y)));
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_21 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(_cvt_f32_43, s_21);
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_cvt_f32_44));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y)));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&_cvt_f32_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float2 r_22 = make_float2(r0_22, r1_22);
        float2 s_22 = fma_f32x2_rn_noftz(r_22, fma_f32x2_rn_noftz(nd_22, r_22, make_float2(1.0f, 1.0f)), r_22);
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(_cvt_f32_45, s_22);
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_cvt_f32_46));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y)));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&_cvt_f32_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_23 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(_cvt_f32_47, s_23);
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_cvt_f32_48));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y)));
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float2 r_24 = make_float2(r0_24, r1_24);
        float2 s_24 = fma_f32x2_rn_noftz(r_24, fma_f32x2_rn_noftz(nd_24, r_24, make_float2(1.0f, 1.0f)), r_24);
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(_cvt_f32_49, s_24);
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_cvt_f32_50));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y)));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float2 _f32x2_mul_b_60 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&_cvt_f32_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_25 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(_cvt_f32_51, s_25);
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_cvt_f32_52));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y)));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&_cvt_f32_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float2 r_26 = make_float2(r0_26, r1_26);
        float2 s_26 = fma_f32x2_rn_noftz(r_26, fma_f32x2_rn_noftz(nd_26, r_26, make_float2(1.0f, 1.0f)), r_26);
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(_cvt_f32_53, s_26);
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_cvt_f32_54));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y)));
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&_cvt_f32_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_27 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(_cvt_f32_55, s_27);
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_cvt_f32_56));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y)));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float2 _f32x2_mul_b_66 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&_cvt_f32_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float2 r_28 = make_float2(r0_28, r1_28);
        float2 s_28 = fma_f32x2_rn_noftz(r_28, fma_f32x2_rn_noftz(nd_28, r_28, make_float2(1.0f, 1.0f)), r_28);
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(_cvt_f32_57, s_28);
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_cvt_f32_58));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y)));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&_cvt_f32_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_29 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(_cvt_f32_59, s_29);
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_cvt_f32_60));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y)));
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float2 _f32x2_mul_b_70 = make_float2(-1.442695f, -1.442695f);
        float2 t_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&_cvt_f32_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float e0_30 = approx_exp2(t_30.x);
        float e1_30 = approx_exp2(t_30.y);
        float2 nd_30 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_30, e1_30));
        float d0_30 = -nd_30.x;
        float d1_30 = -nd_30.y;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float2 r_30 = make_float2(r0_30, r1_30);
        float2 s_30 = fma_f32x2_rn_noftz(r_30, fma_f32x2_rn_noftz(nd_30, r_30, make_float2(1.0f, 1.0f)), r_30);
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(_cvt_f32_61, s_30);
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_cvt_f32_62));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y)));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float2 _f32x2_mul_b_72 = make_float2(-1.442695f, -1.442695f);
        float2 t_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31) : "l"(*(const unsigned long long*)&_cvt_f32_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        float e0_31 = approx_exp2(t_31.x);
        float e1_31 = approx_exp2(t_31.y);
        float2 nd_31 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31, e1_31));
        float d0_31 = -nd_31.x;
        float d1_31 = -nd_31.y;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float2 r_31 = make_float2(r0_31, r1_31);
        float2 s_31 = fma_f32x2_rn_noftz(r_31, fma_f32x2_rn_noftz(nd_31, r_31, make_float2(1.0f, 1.0f)), r_31);
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(_cvt_f32_63, s_31);
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_cvt_f32_64));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y)));
    } else {
        int band = 0;
        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_74 = make_float2(-1.442695f, -1.442695f);
        float2 t_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_32) : "l"(*(const unsigned long long*)&_cvt_f32_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
        float e0_32 = approx_exp2(t_32.x);
        float e1_32 = approx_exp2(t_32.y);
        float2 nd_32 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_32, e1_32));
        float d0_32 = -nd_32.x;
        float d1_32 = -nd_32.y;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float2 r_32 = make_float2(r0_32, r1_32);
        float2 s_32 = fma_f32x2_rn_noftz(r_32, fma_f32x2_rn_noftz(nd_32, r_32, make_float2(1.0f, 1.0f)), r_32);
        float t0_32 = s_32.x;
        float t1_32 = s_32.y;
        band = band | ((fminf(t0_32, t1_32) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(_cvt_f32_65, make_float2(max_noftz(t0_32, 0.0f), max_noftz(t1_32, 0.0f)));
        float2 _mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&_cvt_f32_66));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y)));
        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_76 = make_float2(-1.442695f, -1.442695f);
        float2 t_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_33) : "l"(*(const unsigned long long*)&_cvt_f32_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
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
        band = band | ((fminf(t0_33, t1_33) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(_cvt_f32_67, make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f)));
        float2 _mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_cvt_f32_68));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y)));
        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_78 = make_float2(-1.442695f, -1.442695f);
        float2 t_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_34) : "l"(*(const unsigned long long*)&_cvt_f32_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
        float e0_34 = approx_exp2(t_34.x);
        float e1_34 = approx_exp2(t_34.y);
        float2 nd_34 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_34, e1_34));
        float d0_34 = -nd_34.x;
        float d1_34 = -nd_34.y;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float2 r_34 = make_float2(r0_34, r1_34);
        float2 s_34 = fma_f32x2_rn_noftz(r_34, fma_f32x2_rn_noftz(nd_34, r_34, make_float2(1.0f, 1.0f)), r_34);
        float t0_34 = s_34.x;
        float t1_34 = s_34.y;
        band = band | ((fminf(t0_34, t1_34) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(_cvt_f32_69, make_float2(max_noftz(t0_34, 0.0f), max_noftz(t1_34, 0.0f)));
        float2 _mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_cvt_f32_70));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_34.x, _mul_34.y)));
        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_80 = make_float2(-1.442695f, -1.442695f);
        float2 t_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_35) : "l"(*(const unsigned long long*)&_cvt_f32_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
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
        band = band | ((fminf(t0_35, t1_35) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_81 = mul_f32x2_noftz(_cvt_f32_71, make_float2(max_noftz(t0_35, 0.0f), max_noftz(t1_35, 0.0f)));
        float2 _mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_cvt_f32_72));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_35.x, _mul_35.y)));
        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
        float2 t_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_36) : "l"(*(const unsigned long long*)&_cvt_f32_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
        float e0_36 = approx_exp2(t_36.x);
        float e1_36 = approx_exp2(t_36.y);
        float2 nd_36 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_36, e1_36));
        float d0_36 = -nd_36.x;
        float d1_36 = -nd_36.y;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float2 r_36 = make_float2(r0_36, r1_36);
        float2 s_36 = fma_f32x2_rn_noftz(r_36, fma_f32x2_rn_noftz(nd_36, r_36, make_float2(1.0f, 1.0f)), r_36);
        float t0_36 = s_36.x;
        float t1_36 = s_36.y;
        band = band | ((fminf(t0_36, t1_36) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(_cvt_f32_73, make_float2(max_noftz(t0_36, 0.0f), max_noftz(t1_36, 0.0f)));
        float2 _mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_cvt_f32_74));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_36.x, _mul_36.y)));
        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_84 = make_float2(-1.442695f, -1.442695f);
        float2 t_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_37) : "l"(*(const unsigned long long*)&_cvt_f32_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
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
        band = band | ((fminf(t0_37, t1_37) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(_cvt_f32_75, make_float2(max_noftz(t0_37, 0.0f), max_noftz(t1_37, 0.0f)));
        float2 _mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_cvt_f32_76));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_37.x, _mul_37.y)));
        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_86 = make_float2(-1.442695f, -1.442695f);
        float2 t_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_38) : "l"(*(const unsigned long long*)&_cvt_f32_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
        float e0_38 = approx_exp2(t_38.x);
        float e1_38 = approx_exp2(t_38.y);
        float2 nd_38 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_38, e1_38));
        float d0_38 = -nd_38.x;
        float d1_38 = -nd_38.y;
        float r0_38 = approx_rcp(d0_38);
        float r1_38 = approx_rcp(d1_38);
        float2 r_38 = make_float2(r0_38, r1_38);
        float2 s_38 = fma_f32x2_rn_noftz(r_38, fma_f32x2_rn_noftz(nd_38, r_38, make_float2(1.0f, 1.0f)), r_38);
        float t0_38 = s_38.x;
        float t1_38 = s_38.y;
        band = band | ((fminf(t0_38, t1_38) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(_cvt_f32_77, make_float2(max_noftz(t0_38, 0.0f), max_noftz(t1_38, 0.0f)));
        float2 _mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_cvt_f32_78));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_38.x, _mul_38.y)));
        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_88 = make_float2(-1.442695f, -1.442695f);
        float2 t_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_39) : "l"(*(const unsigned long long*)&_cvt_f32_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
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
        band = band | ((fminf(t0_39, t1_39) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(_cvt_f32_79, make_float2(max_noftz(t0_39, 0.0f), max_noftz(t1_39, 0.0f)));
        float2 _mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_cvt_f32_80));
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
                    float2 _f32x2_mul_b_90 = make_float2(-1.442695f, -1.442695f);
                    float2 t_40;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_40) : "l"(*(const unsigned long long*)&_cvt_f32_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                    float e0_40 = approx_exp2(t_40.x);
                    float e1_40 = approx_exp2(t_40.y);
                    float2 nd_40 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_40, e1_40));
                    float d0_40 = -nd_40.x;
                    float d1_40 = -nd_40.y;
                    float r0_40 = approx_rcp(d0_40);
                    float r1_40 = approx_rcp(d1_40);
                    float2 r_40 = make_float2(r0_40, r1_40);
                    float2 s_40 = fma_f32x2_rn_noftz(r_40, fma_f32x2_rn_noftz(nd_40, r_40, make_float2(1.0f, 1.0f)), r_40);
                    float t0_40 = s_40.x;
                    float t1_40 = s_40.y;
                    float d4_40a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_40a) : "f"(d0_40), "f"(0.25f));
                    float r4_40a = approx_rcp(d4_40a);
                    float _fma_92;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_40a), "f"(r4_40a), "f"(1.0f));
                    float _fma_91;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_40a), "f"(_fma_92), "f"(r4_40a));
                    float big_40a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_40a) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_93;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_40a), "f"(8.388608e+06f));
                    float whole_40a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_40a) : "f"(_add_93), "f"(8.388608e+06f));
                    float above_40a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_40a) : "f"(-d4_40a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_40a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_95;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_40a), "f"(whole_40a));
                    float _abs_94;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
                    float _add_96;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_40a), "f"(0.5f));
                    float _sub_97;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_40a), "f"(0.5f));
                    float rounded_40a = ((_abs_94 == 0.5f) ? ((above_40a > 0.0f) ? _add_96 : _sub_97) : whole_40a);
                    float u0_40 = ((d0_40 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_40a, 7.888609e-31f), 1.7763568e-15f) : t0_40);
                    float d4_40b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_40b) : "f"(d1_40), "f"(0.25f));
                    float r4_40b = approx_rcp(d4_40b);
                    float _fma_99;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_40b), "f"(r4_40b), "f"(1.0f));
                    float _fma_98;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_40b), "f"(_fma_99), "f"(r4_40b));
                    float big_40b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_40b) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_100;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_40b), "f"(8.388608e+06f));
                    float whole_40b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_40b) : "f"(_add_100), "f"(8.388608e+06f));
                    float above_40b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_40b) : "f"(-d4_40b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_40b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_102;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_40b), "f"(whole_40b));
                    float _abs_101;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
                    float _add_103;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_40b), "f"(0.5f));
                    float _sub_104;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_40b), "f"(0.5f));
                    float rounded_40b = ((_abs_101 == 0.5f) ? ((above_40b > 0.0f) ? _add_103 : _sub_104) : whole_40b);
                    float u1_40 = ((d1_40 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_40b, 7.888609e-31f), 1.7763568e-15f) : t1_40);
                    float2 _f32x2_mul_a_105 = mul_f32x2_noftz(_cvt_f32_81, make_float2(max_noftz(u0_40, 0.0f), max_noftz(u1_40, 0.0f)));
                    float2 _mul_40;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_cvt_f32_82));
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
        float2 _f32x2_mul_b_106 = make_float2(-1.442695f, -1.442695f);
        float2 t_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_41) : "l"(*(const unsigned long long*)&_cvt_f32_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
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
        band_0 = band_0 | ((fminf(t0_41, t1_41) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_107 = mul_f32x2_noftz(_cvt_f32_83, make_float2(max_noftz(t0_41, 0.0f), max_noftz(t1_41, 0.0f)));
        float2 _mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_cvt_f32_84));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_41.x, _mul_41.y)));
        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _f32x2_mul_b_108 = make_float2(-1.442695f, -1.442695f);
        float2 t_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_42) : "l"(*(const unsigned long long*)&_cvt_f32_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
        float e0_42 = approx_exp2(t_42.x);
        float e1_42 = approx_exp2(t_42.y);
        float2 nd_42 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_42, e1_42));
        float d0_42 = -nd_42.x;
        float d1_42 = -nd_42.y;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float2 r_42 = make_float2(r0_42, r1_42);
        float2 s_42 = fma_f32x2_rn_noftz(r_42, fma_f32x2_rn_noftz(nd_42, r_42, make_float2(1.0f, 1.0f)), r_42);
        float t0_42 = s_42.x;
        float t1_42 = s_42.y;
        band_0 = band_0 | ((fminf(t0_42, t1_42) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(_cvt_f32_85, make_float2(max_noftz(t0_42, 0.0f), max_noftz(t1_42, 0.0f)));
        float2 _mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_cvt_f32_86));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_42.x, _mul_42.y)));
        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _f32x2_mul_b_110 = make_float2(-1.442695f, -1.442695f);
        float2 t_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_43) : "l"(*(const unsigned long long*)&_cvt_f32_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
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
        band_0 = band_0 | ((fminf(t0_43, t1_43) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(_cvt_f32_87, make_float2(max_noftz(t0_43, 0.0f), max_noftz(t1_43, 0.0f)));
        float2 _mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_cvt_f32_88));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_43.x, _mul_43.y)));
        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _f32x2_mul_b_112 = make_float2(-1.442695f, -1.442695f);
        float2 t_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_44) : "l"(*(const unsigned long long*)&_cvt_f32_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
        float e0_44 = approx_exp2(t_44.x);
        float e1_44 = approx_exp2(t_44.y);
        float2 nd_44 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_44, e1_44));
        float d0_44 = -nd_44.x;
        float d1_44 = -nd_44.y;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float2 r_44 = make_float2(r0_44, r1_44);
        float2 s_44 = fma_f32x2_rn_noftz(r_44, fma_f32x2_rn_noftz(nd_44, r_44, make_float2(1.0f, 1.0f)), r_44);
        float t0_44 = s_44.x;
        float t1_44 = s_44.y;
        band_0 = band_0 | ((fminf(t0_44, t1_44) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(_cvt_f32_89, make_float2(max_noftz(t0_44, 0.0f), max_noftz(t1_44, 0.0f)));
        float2 _mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_cvt_f32_90));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_44.x, _mul_44.y)));
        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _f32x2_mul_b_114 = make_float2(-1.442695f, -1.442695f);
        float2 t_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_45) : "l"(*(const unsigned long long*)&_cvt_f32_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
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
        band_0 = band_0 | ((fminf(t0_45, t1_45) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(_cvt_f32_91, make_float2(max_noftz(t0_45, 0.0f), max_noftz(t1_45, 0.0f)));
        float2 _mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_cvt_f32_92));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_45.x, _mul_45.y)));
        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _f32x2_mul_b_116 = make_float2(-1.442695f, -1.442695f);
        float2 t_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_46) : "l"(*(const unsigned long long*)&_cvt_f32_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
        float e0_46 = approx_exp2(t_46.x);
        float e1_46 = approx_exp2(t_46.y);
        float2 nd_46 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_46, e1_46));
        float d0_46 = -nd_46.x;
        float d1_46 = -nd_46.y;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float2 r_46 = make_float2(r0_46, r1_46);
        float2 s_46 = fma_f32x2_rn_noftz(r_46, fma_f32x2_rn_noftz(nd_46, r_46, make_float2(1.0f, 1.0f)), r_46);
        float t0_46 = s_46.x;
        float t1_46 = s_46.y;
        band_0 = band_0 | ((fminf(t0_46, t1_46) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(_cvt_f32_93, make_float2(max_noftz(t0_46, 0.0f), max_noftz(t1_46, 0.0f)));
        float2 _mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&_cvt_f32_94));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_46.x, _mul_46.y)));
        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _f32x2_mul_b_118 = make_float2(-1.442695f, -1.442695f);
        float2 t_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_47) : "l"(*(const unsigned long long*)&_cvt_f32_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
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
        band_0 = band_0 | ((fminf(t0_47, t1_47) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(_cvt_f32_95, make_float2(max_noftz(t0_47, 0.0f), max_noftz(t1_47, 0.0f)));
        float2 _mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&_cvt_f32_96));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_47.x, _mul_47.y)));
        float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _f32x2_mul_b_120 = make_float2(-1.442695f, -1.442695f);
        float2 t_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_48) : "l"(*(const unsigned long long*)&_cvt_f32_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
        float e0_48 = approx_exp2(t_48.x);
        float e1_48 = approx_exp2(t_48.y);
        float2 nd_48 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_48, e1_48));
        float d0_48 = -nd_48.x;
        float d1_48 = -nd_48.y;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float2 r_48 = make_float2(r0_48, r1_48);
        float2 s_48 = fma_f32x2_rn_noftz(r_48, fma_f32x2_rn_noftz(nd_48, r_48, make_float2(1.0f, 1.0f)), r_48);
        float t0_48 = s_48.x;
        float t1_48 = s_48.y;
        band_0 = band_0 | ((fminf(t0_48, t1_48) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(_cvt_f32_97, make_float2(max_noftz(t0_48, 0.0f), max_noftz(t1_48, 0.0f)));
        float2 _mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_cvt_f32_98));
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
                    float2 _f32x2_mul_b_122 = make_float2(-1.442695f, -1.442695f);
                    float2 t_49;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_49) : "l"(*(const unsigned long long*)&_cvt_f32_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
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
                    float d4_49a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_49a) : "f"(d0_49), "f"(0.25f));
                    float r4_49a = approx_rcp(d4_49a);
                    float _fma_124;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(-d4_49a), "f"(r4_49a), "f"(1.0f));
                    float _fma_123;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(r4_49a), "f"(_fma_124), "f"(r4_49a));
                    float big_49a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_49a) : "f"(mul_f32_rn_noftz(_fma_123, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_125;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(big_49a), "f"(8.388608e+06f));
                    float whole_49a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_49a) : "f"(_add_125), "f"(8.388608e+06f));
                    float above_49a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_49a) : "f"(-d4_49a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_49a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_127;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_127) : "f"(big_49a), "f"(whole_49a));
                    float _abs_126;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(_sub_127));
                    float _add_128;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(big_49a), "f"(0.5f));
                    float _sub_129;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(big_49a), "f"(0.5f));
                    float rounded_49a = ((_abs_126 == 0.5f) ? ((above_49a > 0.0f) ? _add_128 : _sub_129) : whole_49a);
                    float u0_49 = ((d0_49 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_49a, 7.888609e-31f), 1.7763568e-15f) : t0_49);
                    float d4_49b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_49b) : "f"(d1_49), "f"(0.25f));
                    float r4_49b = approx_rcp(d4_49b);
                    float _fma_131;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d4_49b), "f"(r4_49b), "f"(1.0f));
                    float _fma_130;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(r4_49b), "f"(_fma_131), "f"(r4_49b));
                    float big_49b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_49b) : "f"(mul_f32_rn_noftz(_fma_130, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(big_49b), "f"(8.388608e+06f));
                    float whole_49b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_49b) : "f"(_add_132), "f"(8.388608e+06f));
                    float above_49b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_49b) : "f"(-d4_49b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_49b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_134;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_134) : "f"(big_49b), "f"(whole_49b));
                    float _abs_133;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_133) : "f"(_sub_134));
                    float _add_135;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(big_49b), "f"(0.5f));
                    float _sub_136;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(big_49b), "f"(0.5f));
                    float rounded_49b = ((_abs_133 == 0.5f) ? ((above_49b > 0.0f) ? _add_135 : _sub_136) : whole_49b);
                    float u1_49 = ((d1_49 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_49b, 7.888609e-31f), 1.7763568e-15f) : t1_49);
                    float2 _f32x2_mul_a_137 = mul_f32x2_noftz(_cvt_f32_99, make_float2(max_noftz(u0_49, 0.0f), max_noftz(u1_49, 0.0f)));
                    float2 _mul_49;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&_cvt_f32_100));
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
        float2 _f32x2_mul_b_138 = make_float2(-1.442695f, -1.442695f);
        float2 t_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_50) : "l"(*(const unsigned long long*)&_cvt_f32_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
        float e0_50 = approx_exp2(t_50.x);
        float e1_50 = approx_exp2(t_50.y);
        float2 nd_50 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_50, e1_50));
        float d0_50 = -nd_50.x;
        float d1_50 = -nd_50.y;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float2 r_50 = make_float2(r0_50, r1_50);
        float2 s_50 = fma_f32x2_rn_noftz(r_50, fma_f32x2_rn_noftz(nd_50, r_50, make_float2(1.0f, 1.0f)), r_50);
        float t0_50 = s_50.x;
        float t1_50 = s_50.y;
        band_1 = band_1 | ((fminf(t0_50, t1_50) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(_cvt_f32_101, make_float2(max_noftz(t0_50, 0.0f), max_noftz(t1_50, 0.0f)));
        float2 _mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_cvt_f32_102));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_50.x, _mul_50.y)));
        float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_140 = make_float2(-1.442695f, -1.442695f);
        float2 t_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_51) : "l"(*(const unsigned long long*)&_cvt_f32_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
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
        band_1 = band_1 | ((fminf(t0_51, t1_51) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_141 = mul_f32x2_noftz(_cvt_f32_103, make_float2(max_noftz(t0_51, 0.0f), max_noftz(t1_51, 0.0f)));
        float2 _mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&_cvt_f32_104));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_51.x, _mul_51.y)));
        float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_142 = make_float2(-1.442695f, -1.442695f);
        float2 t_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_52) : "l"(*(const unsigned long long*)&_cvt_f32_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
        float e0_52 = approx_exp2(t_52.x);
        float e1_52 = approx_exp2(t_52.y);
        float2 nd_52 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_52, e1_52));
        float d0_52 = -nd_52.x;
        float d1_52 = -nd_52.y;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float2 r_52 = make_float2(r0_52, r1_52);
        float2 s_52 = fma_f32x2_rn_noftz(r_52, fma_f32x2_rn_noftz(nd_52, r_52, make_float2(1.0f, 1.0f)), r_52);
        float t0_52 = s_52.x;
        float t1_52 = s_52.y;
        band_1 = band_1 | ((fminf(t0_52, t1_52) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(_cvt_f32_105, make_float2(max_noftz(t0_52, 0.0f), max_noftz(t1_52, 0.0f)));
        float2 _mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&_cvt_f32_106));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_52.x, _mul_52.y)));
        float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_144 = make_float2(-1.442695f, -1.442695f);
        float2 t_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_53) : "l"(*(const unsigned long long*)&_cvt_f32_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
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
        band_1 = band_1 | ((fminf(t0_53, t1_53) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_145 = mul_f32x2_noftz(_cvt_f32_107, make_float2(max_noftz(t0_53, 0.0f), max_noftz(t1_53, 0.0f)));
        float2 _mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&_cvt_f32_108));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_53.x, _mul_53.y)));
        float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_146 = make_float2(-1.442695f, -1.442695f);
        float2 t_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_54) : "l"(*(const unsigned long long*)&_cvt_f32_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
        float e0_54 = approx_exp2(t_54.x);
        float e1_54 = approx_exp2(t_54.y);
        float2 nd_54 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_54, e1_54));
        float d0_54 = -nd_54.x;
        float d1_54 = -nd_54.y;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float2 r_54 = make_float2(r0_54, r1_54);
        float2 s_54 = fma_f32x2_rn_noftz(r_54, fma_f32x2_rn_noftz(nd_54, r_54, make_float2(1.0f, 1.0f)), r_54);
        float t0_54 = s_54.x;
        float t1_54 = s_54.y;
        band_1 = band_1 | ((fminf(t0_54, t1_54) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_147 = mul_f32x2_noftz(_cvt_f32_109, make_float2(max_noftz(t0_54, 0.0f), max_noftz(t1_54, 0.0f)));
        float2 _mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&_cvt_f32_110));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_54.x, _mul_54.y)));
        float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_148 = make_float2(-1.442695f, -1.442695f);
        float2 t_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_55) : "l"(*(const unsigned long long*)&_cvt_f32_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_148));
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
        band_1 = band_1 | ((fminf(t0_55, t1_55) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_149 = mul_f32x2_noftz(_cvt_f32_111, make_float2(max_noftz(t0_55, 0.0f), max_noftz(t1_55, 0.0f)));
        float2 _mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&_cvt_f32_112));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_55.x, _mul_55.y)));
        float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_150 = make_float2(-1.442695f, -1.442695f);
        float2 t_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_56) : "l"(*(const unsigned long long*)&_cvt_f32_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_150));
        float e0_56 = approx_exp2(t_56.x);
        float e1_56 = approx_exp2(t_56.y);
        float2 nd_56 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_56, e1_56));
        float d0_56 = -nd_56.x;
        float d1_56 = -nd_56.y;
        float r0_56 = approx_rcp(d0_56);
        float r1_56 = approx_rcp(d1_56);
        float2 r_56 = make_float2(r0_56, r1_56);
        float2 s_56 = fma_f32x2_rn_noftz(r_56, fma_f32x2_rn_noftz(nd_56, r_56, make_float2(1.0f, 1.0f)), r_56);
        float t0_56 = s_56.x;
        float t1_56 = s_56.y;
        band_1 = band_1 | ((fminf(t0_56, t1_56) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_151 = mul_f32x2_noftz(_cvt_f32_113, make_float2(max_noftz(t0_56, 0.0f), max_noftz(t1_56, 0.0f)));
        float2 _mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&_cvt_f32_114));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_56.x, _mul_56.y)));
        float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_152 = make_float2(-1.442695f, -1.442695f);
        float2 t_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_57) : "l"(*(const unsigned long long*)&_cvt_f32_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_152));
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
        band_1 = band_1 | ((fminf(t0_57, t1_57) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(_cvt_f32_115, make_float2(max_noftz(t0_57, 0.0f), max_noftz(t1_57, 0.0f)));
        float2 _mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_cvt_f32_116));
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
                    float2 _f32x2_mul_b_154 = make_float2(-1.442695f, -1.442695f);
                    float2 t_58;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_58) : "l"(*(const unsigned long long*)&_cvt_f32_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
                    float e0_58 = approx_exp2(t_58.x);
                    float e1_58 = approx_exp2(t_58.y);
                    float2 nd_58 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_58, e1_58));
                    float d0_58 = -nd_58.x;
                    float d1_58 = -nd_58.y;
                    float r0_58 = approx_rcp(d0_58);
                    float r1_58 = approx_rcp(d1_58);
                    float2 r_58 = make_float2(r0_58, r1_58);
                    float2 s_58 = fma_f32x2_rn_noftz(r_58, fma_f32x2_rn_noftz(nd_58, r_58, make_float2(1.0f, 1.0f)), r_58);
                    float t0_58 = s_58.x;
                    float t1_58 = s_58.y;
                    float d4_58a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_58a) : "f"(d0_58), "f"(0.25f));
                    float r4_58a = approx_rcp(d4_58a);
                    float _fma_156;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d4_58a), "f"(r4_58a), "f"(1.0f));
                    float _fma_155;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(r4_58a), "f"(_fma_156), "f"(r4_58a));
                    float big_58a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_58a) : "f"(mul_f32_rn_noftz(_fma_155, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_157;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(big_58a), "f"(8.388608e+06f));
                    float whole_58a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_58a) : "f"(_add_157), "f"(8.388608e+06f));
                    float above_58a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_58a) : "f"(-d4_58a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_58a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_159;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_159) : "f"(big_58a), "f"(whole_58a));
                    float _abs_158;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(_sub_159));
                    float _add_160;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(big_58a), "f"(0.5f));
                    float _sub_161;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(big_58a), "f"(0.5f));
                    float rounded_58a = ((_abs_158 == 0.5f) ? ((above_58a > 0.0f) ? _add_160 : _sub_161) : whole_58a);
                    float u0_58 = ((d0_58 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_58a, 7.888609e-31f), 1.7763568e-15f) : t0_58);
                    float d4_58b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_58b) : "f"(d1_58), "f"(0.25f));
                    float r4_58b = approx_rcp(d4_58b);
                    float _fma_163;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(-d4_58b), "f"(r4_58b), "f"(1.0f));
                    float _fma_162;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(r4_58b), "f"(_fma_163), "f"(r4_58b));
                    float big_58b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_58b) : "f"(mul_f32_rn_noftz(_fma_162, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_164;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(big_58b), "f"(8.388608e+06f));
                    float whole_58b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_58b) : "f"(_add_164), "f"(8.388608e+06f));
                    float above_58b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_58b) : "f"(-d4_58b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_58b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_166;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_166) : "f"(big_58b), "f"(whole_58b));
                    float _abs_165;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_165) : "f"(_sub_166));
                    float _add_167;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(big_58b), "f"(0.5f));
                    float _sub_168;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(big_58b), "f"(0.5f));
                    float rounded_58b = ((_abs_165 == 0.5f) ? ((above_58b > 0.0f) ? _add_167 : _sub_168) : whole_58b);
                    float u1_58 = ((d1_58 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_58b, 7.888609e-31f), 1.7763568e-15f) : t1_58);
                    float2 _f32x2_mul_a_169 = mul_f32x2_noftz(_cvt_f32_117, make_float2(max_noftz(u0_58, 0.0f), max_noftz(u1_58, 0.0f)));
                    float2 _mul_58;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&_cvt_f32_118));
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
        float2 _f32x2_mul_b_170 = make_float2(-1.442695f, -1.442695f);
        float2 t_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_59) : "l"(*(const unsigned long long*)&_cvt_f32_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_170));
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
        band_2 = band_2 | ((fminf(t0_59, t1_59) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_171 = mul_f32x2_noftz(_cvt_f32_119, make_float2(max_noftz(t0_59, 0.0f), max_noftz(t1_59, 0.0f)));
        float2 _mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&_cvt_f32_120));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_59.x, _mul_59.y)));
        float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float2 _f32x2_mul_b_172 = make_float2(-1.442695f, -1.442695f);
        float2 t_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_60) : "l"(*(const unsigned long long*)&_cvt_f32_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_172));
        float e0_60 = approx_exp2(t_60.x);
        float e1_60 = approx_exp2(t_60.y);
        float2 nd_60 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_60, e1_60));
        float d0_60 = -nd_60.x;
        float d1_60 = -nd_60.y;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float2 r_60 = make_float2(r0_60, r1_60);
        float2 s_60 = fma_f32x2_rn_noftz(r_60, fma_f32x2_rn_noftz(nd_60, r_60, make_float2(1.0f, 1.0f)), r_60);
        float t0_60 = s_60.x;
        float t1_60 = s_60.y;
        band_2 = band_2 | ((fminf(t0_60, t1_60) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(_cvt_f32_121, make_float2(max_noftz(t0_60, 0.0f), max_noftz(t1_60, 0.0f)));
        float2 _mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&_cvt_f32_122));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_60.x, _mul_60.y)));
        float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float2 _f32x2_mul_b_174 = make_float2(-1.442695f, -1.442695f);
        float2 t_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_61) : "l"(*(const unsigned long long*)&_cvt_f32_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_174));
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
        band_2 = band_2 | ((fminf(t0_61, t1_61) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_175 = mul_f32x2_noftz(_cvt_f32_123, make_float2(max_noftz(t0_61, 0.0f), max_noftz(t1_61, 0.0f)));
        float2 _mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&_cvt_f32_124));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_61.x, _mul_61.y)));
        float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float2 _f32x2_mul_b_176 = make_float2(-1.442695f, -1.442695f);
        float2 t_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_62) : "l"(*(const unsigned long long*)&_cvt_f32_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
        float e0_62 = approx_exp2(t_62.x);
        float e1_62 = approx_exp2(t_62.y);
        float2 nd_62 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_62, e1_62));
        float d0_62 = -nd_62.x;
        float d1_62 = -nd_62.y;
        float r0_62 = approx_rcp(d0_62);
        float r1_62 = approx_rcp(d1_62);
        float2 r_62 = make_float2(r0_62, r1_62);
        float2 s_62 = fma_f32x2_rn_noftz(r_62, fma_f32x2_rn_noftz(nd_62, r_62, make_float2(1.0f, 1.0f)), r_62);
        float t0_62 = s_62.x;
        float t1_62 = s_62.y;
        band_2 = band_2 | ((fminf(t0_62, t1_62) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_177 = mul_f32x2_noftz(_cvt_f32_125, make_float2(max_noftz(t0_62, 0.0f), max_noftz(t1_62, 0.0f)));
        float2 _mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&_cvt_f32_126));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_62.x, _mul_62.y)));
        float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_128 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float2 _f32x2_mul_b_178 = make_float2(-1.442695f, -1.442695f);
        float2 t_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_63) : "l"(*(const unsigned long long*)&_cvt_f32_127), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
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
        band_2 = band_2 | ((fminf(t0_63, t1_63) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_179 = mul_f32x2_noftz(_cvt_f32_127, make_float2(max_noftz(t0_63, 0.0f), max_noftz(t1_63, 0.0f)));
        float2 _mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&_cvt_f32_128));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_63.x, _mul_63.y)));
        float2 _cvt_f32_129 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_130 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float2 _f32x2_mul_b_180 = make_float2(-1.442695f, -1.442695f);
        float2 t_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_64) : "l"(*(const unsigned long long*)&_cvt_f32_129), "l"(*(const unsigned long long*)&_f32x2_mul_b_180));
        float e0_64 = approx_exp2(t_64.x);
        float e1_64 = approx_exp2(t_64.y);
        float2 nd_64 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_64, e1_64));
        float d0_64 = -nd_64.x;
        float d1_64 = -nd_64.y;
        float r0_64 = approx_rcp(d0_64);
        float r1_64 = approx_rcp(d1_64);
        float2 r_64 = make_float2(r0_64, r1_64);
        float2 s_64 = fma_f32x2_rn_noftz(r_64, fma_f32x2_rn_noftz(nd_64, r_64, make_float2(1.0f, 1.0f)), r_64);
        float t0_64 = s_64.x;
        float t1_64 = s_64.y;
        band_2 = band_2 | ((fminf(t0_64, t1_64) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_181 = mul_f32x2_noftz(_cvt_f32_129, make_float2(max_noftz(t0_64, 0.0f), max_noftz(t1_64, 0.0f)));
        float2 _mul_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&_cvt_f32_130));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_64.x, _mul_64.y)));
        float2 _cvt_f32_131 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_132 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float2 _f32x2_mul_b_182 = make_float2(-1.442695f, -1.442695f);
        float2 t_65;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_65) : "l"(*(const unsigned long long*)&_cvt_f32_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
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
        band_2 = band_2 | ((fminf(t0_65, t1_65) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_183 = mul_f32x2_noftz(_cvt_f32_131, make_float2(max_noftz(t0_65, 0.0f), max_noftz(t1_65, 0.0f)));
        float2 _mul_65;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&_cvt_f32_132));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_65.x, _mul_65.y)));
        float2 _cvt_f32_133 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_134 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float2 _f32x2_mul_b_184 = make_float2(-1.442695f, -1.442695f);
        float2 t_66;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_66) : "l"(*(const unsigned long long*)&_cvt_f32_133), "l"(*(const unsigned long long*)&_f32x2_mul_b_184));
        float e0_66 = approx_exp2(t_66.x);
        float e1_66 = approx_exp2(t_66.y);
        float2 nd_66 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_66, e1_66));
        float d0_66 = -nd_66.x;
        float d1_66 = -nd_66.y;
        float r0_66 = approx_rcp(d0_66);
        float r1_66 = approx_rcp(d1_66);
        float2 r_66 = make_float2(r0_66, r1_66);
        float2 s_66 = fma_f32x2_rn_noftz(r_66, fma_f32x2_rn_noftz(nd_66, r_66, make_float2(1.0f, 1.0f)), r_66);
        float t0_66 = s_66.x;
        float t1_66 = s_66.y;
        band_2 = band_2 | ((fminf(t0_66, t1_66) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_185 = mul_f32x2_noftz(_cvt_f32_133, make_float2(max_noftz(t0_66, 0.0f), max_noftz(t1_66, 0.0f)));
        float2 _mul_66;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&_cvt_f32_134));
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
                    float2 _f32x2_mul_b_186 = make_float2(-1.442695f, -1.442695f);
                    float2 t_67;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_67) : "l"(*(const unsigned long long*)&_cvt_f32_135), "l"(*(const unsigned long long*)&_f32x2_mul_b_186));
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
                    float _fma_188;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(-d4_67a), "f"(r4_67a), "f"(1.0f));
                    float _fma_187;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(r4_67a), "f"(_fma_188), "f"(r4_67a));
                    float big_67a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67a) : "f"(mul_f32_rn_noftz(_fma_187, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_189;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(big_67a), "f"(8.388608e+06f));
                    float whole_67a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67a) : "f"(_add_189), "f"(8.388608e+06f));
                    float above_67a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67a) : "f"(-d4_67a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_191;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_191) : "f"(big_67a), "f"(whole_67a));
                    float _abs_190;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_190) : "f"(_sub_191));
                    float _add_192;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(big_67a), "f"(0.5f));
                    float _sub_193;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_193) : "f"(big_67a), "f"(0.5f));
                    float rounded_67a = ((_abs_190 == 0.5f) ? ((above_67a > 0.0f) ? _add_192 : _sub_193) : whole_67a);
                    float u0_67 = ((d0_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67a, 7.888609e-31f), 1.7763568e-15f) : t0_67);
                    float d4_67b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67b) : "f"(d1_67), "f"(0.25f));
                    float r4_67b = approx_rcp(d4_67b);
                    float _fma_195;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d4_67b), "f"(r4_67b), "f"(1.0f));
                    float _fma_194;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(r4_67b), "f"(_fma_195), "f"(r4_67b));
                    float big_67b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67b) : "f"(mul_f32_rn_noftz(_fma_194, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_196;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(big_67b), "f"(8.388608e+06f));
                    float whole_67b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67b) : "f"(_add_196), "f"(8.388608e+06f));
                    float above_67b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67b) : "f"(-d4_67b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_198;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_198) : "f"(big_67b), "f"(whole_67b));
                    float _abs_197;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_197) : "f"(_sub_198));
                    float _add_199;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(big_67b), "f"(0.5f));
                    float _sub_200;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_200) : "f"(big_67b), "f"(0.5f));
                    float rounded_67b = ((_abs_197 == 0.5f) ? ((above_67b > 0.0f) ? _add_199 : _sub_200) : whole_67b);
                    float u1_67 = ((d1_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67b, 7.888609e-31f), 1.7763568e-15f) : t1_67);
                    float2 _f32x2_mul_a_201 = mul_f32x2_noftz(_cvt_f32_135, make_float2(max_noftz(u0_67, 0.0f), max_noftz(u1_67, 0.0f)));
                    float2 _mul_67;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&_cvt_f32_136));
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
    __nv_bfloat162 _hxa_203 = amax_a;
    __nv_bfloat162 _hxb_203 = __as_bf16x2(cols[0]);
    unsigned int _hxr_203;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_203) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_203)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_203)));
    __nv_bfloat162 _max_202 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_203);
    amax_a = _max_202;
    __nv_bfloat162 _hxa_205 = amax_b;
    __nv_bfloat162 _hxb_205 = __as_bf16x2(cols[16]);
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
    __nv_bfloat162 _hxb_209 = __as_bf16x2(cols[17]);
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
    __nv_bfloat162 _hxb_213 = __as_bf16x2(cols[18]);
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
    __nv_bfloat162 _hxb_217 = __as_bf16x2(cols[19]);
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
    __nv_bfloat162 _hxb_221 = __as_bf16x2(cols[20]);
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
    __nv_bfloat162 _hxb_225 = __as_bf16x2(cols[21]);
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
    __nv_bfloat162 _hxb_229 = __as_bf16x2(cols[22]);
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
    __nv_bfloat162 _hxb_233 = __as_bf16x2(cols[23]);
    unsigned int _hxr_233;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_233) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_233)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_233)));
    __nv_bfloat162 _max_232 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_233);
    amax_b = _max_232;
    __nv_bfloat162 _hxa_235 = amax_a;
    __nv_bfloat162 _hxb_235 = __as_bf16x2(cols[8]);
    unsigned int _hxr_235;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_235) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_235)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_235)));
    __nv_bfloat162 _max_234 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_235);
    amax_a = _max_234;
    __nv_bfloat162 _hxa_237 = amax_b;
    __nv_bfloat162 _hxb_237 = __as_bf16x2(cols[24]);
    unsigned int _hxr_237;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_237) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_237)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_237)));
    __nv_bfloat162 _max_236 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_237);
    amax_b = _max_236;
    __nv_bfloat162 _hxa_239 = amax_a;
    __nv_bfloat162 _hxb_239 = __as_bf16x2(cols[9]);
    unsigned int _hxr_239;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_239) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_239)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_239)));
    __nv_bfloat162 _max_238 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_239);
    amax_a = _max_238;
    __nv_bfloat162 _hxa_241 = amax_b;
    __nv_bfloat162 _hxb_241 = __as_bf16x2(cols[25]);
    unsigned int _hxr_241;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_241) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_241)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_241)));
    __nv_bfloat162 _max_240 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_241);
    amax_b = _max_240;
    __nv_bfloat162 _hxa_243 = amax_a;
    __nv_bfloat162 _hxb_243 = __as_bf16x2(cols[10]);
    unsigned int _hxr_243;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_243) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_243)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_243)));
    __nv_bfloat162 _max_242 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_243);
    amax_a = _max_242;
    __nv_bfloat162 _hxa_245 = amax_b;
    __nv_bfloat162 _hxb_245 = __as_bf16x2(cols[26]);
    unsigned int _hxr_245;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_245) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_245)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_245)));
    __nv_bfloat162 _max_244 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_245);
    amax_b = _max_244;
    __nv_bfloat162 _hxa_247 = amax_a;
    __nv_bfloat162 _hxb_247 = __as_bf16x2(cols[11]);
    unsigned int _hxr_247;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_247) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_247)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_247)));
    __nv_bfloat162 _max_246 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_247);
    amax_a = _max_246;
    __nv_bfloat162 _hxa_249 = amax_b;
    __nv_bfloat162 _hxb_249 = __as_bf16x2(cols[27]);
    unsigned int _hxr_249;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_249) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_249)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_249)));
    __nv_bfloat162 _max_248 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_249);
    amax_b = _max_248;
    __nv_bfloat162 _hxa_251 = amax_a;
    __nv_bfloat162 _hxb_251 = __as_bf16x2(cols[12]);
    unsigned int _hxr_251;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_251) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_251)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_251)));
    __nv_bfloat162 _max_250 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_251);
    amax_a = _max_250;
    __nv_bfloat162 _hxa_253 = amax_b;
    __nv_bfloat162 _hxb_253 = __as_bf16x2(cols[28]);
    unsigned int _hxr_253;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_253) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_253)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_253)));
    __nv_bfloat162 _max_252 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_253);
    amax_b = _max_252;
    __nv_bfloat162 _hxa_255 = amax_a;
    __nv_bfloat162 _hxb_255 = __as_bf16x2(cols[13]);
    unsigned int _hxr_255;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_255) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_255)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_255)));
    __nv_bfloat162 _max_254 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_255);
    amax_a = _max_254;
    __nv_bfloat162 _hxa_257 = amax_b;
    __nv_bfloat162 _hxb_257 = __as_bf16x2(cols[29]);
    unsigned int _hxr_257;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_257) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_257)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_257)));
    __nv_bfloat162 _max_256 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_257);
    amax_b = _max_256;
    __nv_bfloat162 _hxa_259 = amax_a;
    __nv_bfloat162 _hxb_259 = __as_bf16x2(cols[14]);
    unsigned int _hxr_259;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_259) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_259)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_259)));
    __nv_bfloat162 _max_258 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_259);
    amax_a = _max_258;
    __nv_bfloat162 _hxa_261 = amax_b;
    __nv_bfloat162 _hxb_261 = __as_bf16x2(cols[30]);
    unsigned int _hxr_261;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_261) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_261)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_261)));
    __nv_bfloat162 _max_260 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_261);
    amax_b = _max_260;
    __nv_bfloat162 _hxa_263 = amax_a;
    __nv_bfloat162 _hxb_263 = __as_bf16x2(cols[15]);
    unsigned int _hxr_263;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_263) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_263)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_263)));
    __nv_bfloat162 _max_262 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_263);
    amax_a = _max_262;
    __nv_bfloat162 _hxa_265 = amax_b;
    __nv_bfloat162 _hxb_265 = __as_bf16x2(cols[31]);
    unsigned int _hxr_265;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_265) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_265)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_265)));
    __nv_bfloat162 _max_264 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_265);
    amax_b = _max_264;
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
    unsigned int amax_bits_3 = (col_max_bits >> 16 & 65535) << 16;
    int scale_i32_4 = (int)(amax_bits_3 + 2031616 >> 23) - 8;
    if (scale_i32_4 < 0) {
        scale_i32_4 = 0;
    }
    if ((amax_bits_3 & 2139095040) == 2139095040) {
        scale_i32_4 = 255;
    }
    unsigned int scale_5 = (unsigned int)scale_i32_4;
    unsigned int inv_6 = 254 - scale_5 << 7;
    if (scale_5 == 255) {
        inv_6 = 32704;
    }
    unsigned int inv2_7 = inv_6 | inv_6 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_267;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_266;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_266) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_267));
    words_a[0] = _mov_b32_pack_u16x2_266;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(cols[16]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(cols[17]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_269;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_269) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_269) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_268;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_268) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_269));
    words_b[0] = _mov_b32_pack_u16x2_268;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_271;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_271) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_271) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_270;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_270) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_271));
    words_a[1] = _mov_b32_pack_u16x2_270;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(cols[18]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(cols[19]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_273;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_273) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_273) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_272;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_272) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_273));
    words_b[1] = _mov_b32_pack_u16x2_272;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_275;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_275) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_275) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_274;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_274) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_275));
    words_a[2] = _mov_b32_pack_u16x2_274;
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(cols[20]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(cols[21]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_277;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_277) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_277) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_276;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_276) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_277));
    words_b[2] = _mov_b32_pack_u16x2_276;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_279;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_279) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_279) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_278;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_278) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_279));
    words_a[3] = _mov_b32_pack_u16x2_278;
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(cols[22]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(cols[23]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_281;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_281) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_281) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_280;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_280) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_281));
    words_b[3] = _mov_b32_pack_u16x2_280;
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(cols[8]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(cols[9]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_283;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_282;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_282) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_283));
    words_a[4] = _mov_b32_pack_u16x2_282;
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(cols[24]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(cols[25]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_285;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_284;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_284) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_285));
    words_b[4] = _mov_b32_pack_u16x2_284;
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(cols[10]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(cols[11]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_287;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_286;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_286) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_287));
    words_a[5] = _mov_b32_pack_u16x2_286;
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(cols[26]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(cols[27]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_289;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_288;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_288) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_289));
    words_b[5] = _mov_b32_pack_u16x2_288;
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(cols[12]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(cols[13]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_291;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_290;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_290) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_291));
    words_a[6] = _mov_b32_pack_u16x2_290;
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(cols[28]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(cols[29]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_293;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_292;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_292) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_293));
    words_b[6] = _mov_b32_pack_u16x2_292;
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(cols[14]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(cols[15]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_295;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_294;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_294) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_295));
    words_a[7] = _mov_b32_pack_u16x2_294;
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(cols[30]), "r"(inv2_7));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(cols[31]), "r"(inv2_7));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_297;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_296;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_296) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_297));
    words_b[7] = _mov_b32_pack_u16x2_296;
    {
        const unsigned* _raw_stv8_298 = reinterpret_cast<const unsigned*>(words_a + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a * M + out_row))), "r"(_raw_stv8_298[0]), "r"(_raw_stv8_298[1]), "r"(_raw_stv8_298[2]), "r"(_raw_stv8_298[3]), "r"(_raw_stv8_298[4]), "r"(_raw_stv8_298[5]), "r"(_raw_stv8_298[6]), "r"(_raw_stv8_298[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_299 = reinterpret_cast<const unsigned*>(words_b + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b * M + out_row))), "r"(_raw_stv8_299[0]), "r"(_raw_stv8_299[1]), "r"(_raw_stv8_299[2]), "r"(_raw_stv8_299[3]), "r"(_raw_stv8_299[4]), "r"(_raw_stv8_299[5]), "r"(_raw_stv8_299[6]), "r"(_raw_stv8_299[7]) : "memory");
    }
    int row_tiles = M >> 7;
    int rblk = by * 4 + (lane_1 & 3);
    int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
    int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_5);
}

} // extern "C"
