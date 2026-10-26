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
#define SMEM_TILES_STAGE_BYTES 32768
#define SMEM_TILES_STRIDE 32768
#define SMEM_TOTAL 32768
#define THREADS 128
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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_49303287790d1f318420(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 128 + row;
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
    int grow_b = grow + 32;
    int grow_c = grow + 64;
    int grow_d = grow + 96;
    int gate_index_0 = grow_b * (2 * K) + col;
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_0 + (0))) : "memory");
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_0 + K) + (0))) : "memory");
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
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(grad + (grow_b * K + col) + (0))) : "memory");
        _vec_load_5[0 + 0] = (uint32_t)_iv_5_0;
        _vec_load_5[0 + 1] = (uint32_t)_iv_5_1;
        _vec_load_5[0 + 2] = (uint32_t)_iv_5_2;
        _vec_load_5[0 + 3] = (uint32_t)_iv_5_3;
        _vec_load_5[0 + 4] = (uint32_t)_iv_5_4;
        _vec_load_5[0 + 5] = (uint32_t)_iv_5_5;
        _vec_load_5[0 + 6] = (uint32_t)_iv_5_6;
        _vec_load_5[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_1 = grow_c * (2 * K) + col;
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
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_1 + (0))) : "memory");
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
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_1 + K) + (0))) : "memory");
        _vec_load_7[0 + 0] = (uint32_t)_iv_7_0;
        _vec_load_7[0 + 1] = (uint32_t)_iv_7_1;
        _vec_load_7[0 + 2] = (uint32_t)_iv_7_2;
        _vec_load_7[0 + 3] = (uint32_t)_iv_7_3;
        _vec_load_7[0 + 4] = (uint32_t)_iv_7_4;
        _vec_load_7[0 + 5] = (uint32_t)_iv_7_5;
        _vec_load_7[0 + 6] = (uint32_t)_iv_7_6;
        _vec_load_7[0 + 7] = (uint32_t)_iv_7_7;
    }
    unsigned int _vec_load_8[8];
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
        _vec_load_8[0 + 0] = (uint32_t)_iv_8_0;
        _vec_load_8[0 + 1] = (uint32_t)_iv_8_1;
        _vec_load_8[0 + 2] = (uint32_t)_iv_8_2;
        _vec_load_8[0 + 3] = (uint32_t)_iv_8_3;
        _vec_load_8[0 + 4] = (uint32_t)_iv_8_4;
        _vec_load_8[0 + 5] = (uint32_t)_iv_8_5;
        _vec_load_8[0 + 6] = (uint32_t)_iv_8_6;
        _vec_load_8[0 + 7] = (uint32_t)_iv_8_7;
    }
    int gate_index_2 = grow_d * (2 * K) + col;
    unsigned int _vec_load_9[8];
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
        _vec_load_9[0 + 0] = (uint32_t)_iv_9_0;
        _vec_load_9[0 + 1] = (uint32_t)_iv_9_1;
        _vec_load_9[0 + 2] = (uint32_t)_iv_9_2;
        _vec_load_9[0 + 3] = (uint32_t)_iv_9_3;
        _vec_load_9[0 + 4] = (uint32_t)_iv_9_4;
        _vec_load_9[0 + 5] = (uint32_t)_iv_9_5;
        _vec_load_9[0 + 6] = (uint32_t)_iv_9_6;
        _vec_load_9[0 + 7] = (uint32_t)_iv_9_7;
    }
    unsigned int _vec_load_10[8];
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
        _vec_load_10[0 + 0] = (uint32_t)_iv_10_0;
        _vec_load_10[0 + 1] = (uint32_t)_iv_10_1;
        _vec_load_10[0 + 2] = (uint32_t)_iv_10_2;
        _vec_load_10[0 + 3] = (uint32_t)_iv_10_3;
        _vec_load_10[0 + 4] = (uint32_t)_iv_10_4;
        _vec_load_10[0 + 5] = (uint32_t)_iv_10_5;
        _vec_load_10[0 + 6] = (uint32_t)_iv_10_6;
        _vec_load_10[0 + 7] = (uint32_t)_iv_10_7;
    }
    unsigned int _vec_load_11[8];
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
        _vec_load_11[0 + 0] = (uint32_t)_iv_11_0;
        _vec_load_11[0 + 1] = (uint32_t)_iv_11_1;
        _vec_load_11[0 + 2] = (uint32_t)_iv_11_2;
        _vec_load_11[0 + 3] = (uint32_t)_iv_11_3;
        _vec_load_11[0 + 4] = (uint32_t)_iv_11_4;
        _vec_load_11[0 + 5] = (uint32_t)_iv_11_5;
        _vec_load_11[0 + 6] = (uint32_t)_iv_11_6;
        _vec_load_11[0 + 7] = (uint32_t)_iv_11_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_b[8];
    unsigned int dup_b[8];
    unsigned int dgate_c[8];
    unsigned int dup_c[8];
    unsigned int dgate_d[8];
    unsigned int dup_d[8];
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
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_6[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_9[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_13;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_12 = __all_sync(0xFFFFFFFF, _min_13 > -87.0f);
    if (_warp_vote_12 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_1 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0_1), "f"(_fma_15), "f"(r0_1));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r1_1), "f"(_fma_17), "f"(r1_1));
        float2 s_0 = make_float2(_fma_14, _fma_16);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), _cvt_f32_3);
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_cvt_f32_2));
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&_cvt_f32_3));
        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y));
        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y));
        dgate[0] = __as_u32(_bf16x2_0);
        dup[0] = __as_u32(_bf16x2_1);
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_3 = approx_exp2((-_cvt_f32_4.x) * 1.442695f);
        float e1_3 = approx_exp2((-_cvt_f32_4.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r0_3), "f"(_fma_20), "f"(r0_3));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r1_3), "f"(_fma_22), "f"(r1_3));
        float2 s_2 = make_float2(_fma_19, _fma_21);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), _cvt_f32_6);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&_cvt_f32_5));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&_cvt_f32_6));
        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y));
        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y));
        dgate[1] = __as_u32(_bf16x2_2);
        dup[1] = __as_u32(_bf16x2_3);
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_5 = approx_exp2((-_cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-_cvt_f32_7.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_5), "f"(_fma_25), "f"(r0_5));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r1_5), "f"(_fma_27), "f"(r1_5));
        float2 s_4 = make_float2(_fma_24, _fma_26);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), _cvt_f32_9);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_cvt_f32_8));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&_cvt_f32_9));
        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y));
        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y));
        dgate[2] = __as_u32(_bf16x2_4);
        dup[2] = __as_u32(_bf16x2_5);
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_7 = approx_exp2((-_cvt_f32_10.x) * 1.442695f);
        float e1_7 = approx_exp2((-_cvt_f32_10.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r0_7), "f"(_fma_30), "f"(r0_7));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r1_7), "f"(_fma_32), "f"(r1_7));
        float2 s_6 = make_float2(_fma_29, _fma_31);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), _cvt_f32_12);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_cvt_f32_11));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&_cvt_f32_12));
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y));
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y));
        dgate[3] = __as_u32(_bf16x2_6);
        dup[3] = __as_u32(_bf16x2_7);
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_9 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_9), "f"(_fma_35), "f"(r0_9));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r1_9), "f"(_fma_37), "f"(r1_9));
        float2 s_8 = make_float2(_fma_34, _fma_36);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_cvt_f32_14));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&_cvt_f32_15));
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y));
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y));
        dgate[4] = __as_u32(_bf16x2_8);
        dup[4] = __as_u32(_bf16x2_9);
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_11 = approx_exp2((-_cvt_f32_16.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_16.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r0_11), "f"(_fma_40), "f"(r0_11));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r1_11), "f"(_fma_42), "f"(r1_11));
        float2 s_10 = make_float2(_fma_39, _fma_41);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_cvt_f32_17));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&_cvt_f32_18));
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y));
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y));
        dgate[5] = __as_u32(_bf16x2_10);
        dup[5] = __as_u32(_bf16x2_11);
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_13 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(r0_13), "f"(_fma_45), "f"(r0_13));
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r1_13), "f"(_fma_47), "f"(r1_13));
        float2 s_12 = make_float2(_fma_44, _fma_46);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_48 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_cvt_f32_20));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&_cvt_f32_21));
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y));
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y));
        dgate[6] = __as_u32(_bf16x2_12);
        dup[6] = __as_u32(_bf16x2_13);
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_15 = approx_exp2((-_cvt_f32_22.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_22.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(r0_15), "f"(_fma_50), "f"(r0_15));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(r1_15), "f"(_fma_52), "f"(r1_15));
        float2 s_14 = make_float2(_fma_49, _fma_51);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[7] = __as_u32(_bf16x2_14);
        dup[7] = __as_u32(_bf16x2_15);
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_17 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_17 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(r0_17), "f"(_fma_55), "f"(r0_17));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r1_17), "f"(_fma_57), "f"(r1_17));
        float2 s_16 = make_float2(_fma_54, _fma_56);
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), _cvt_f32_27);
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_cvt_f32_26));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&_cvt_f32_27));
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y));
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y));
        dgate_b[0] = __as_u32(_bf16x2_16);
        dup_b[0] = __as_u32(_bf16x2_17);
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_19 = approx_exp2((-_cvt_f32_28.x) * 1.442695f);
        float e1_19 = approx_exp2((-_cvt_f32_28.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(r0_19), "f"(_fma_60), "f"(r0_19));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(r1_19), "f"(_fma_62), "f"(r1_19));
        float2 s_18 = make_float2(_fma_59, _fma_61);
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&_cvt_f32_28), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), _cvt_f32_30);
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_cvt_f32_29));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&_cvt_f32_30));
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y));
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y));
        dgate_b[1] = __as_u32(_bf16x2_18);
        dup_b[1] = __as_u32(_bf16x2_19);
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_21 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_21 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r0_21), "f"(_fma_65), "f"(r0_21));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r1_21), "f"(_fma_67), "f"(r1_21));
        float2 s_20 = make_float2(_fma_64, _fma_66);
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), _cvt_f32_33);
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_cvt_f32_32));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&_cvt_f32_33));
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y));
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y));
        dgate_b[2] = __as_u32(_bf16x2_20);
        dup_b[2] = __as_u32(_bf16x2_21);
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_23 = approx_exp2((-_cvt_f32_34.x) * 1.442695f);
        float e1_23 = approx_exp2((-_cvt_f32_34.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(r0_23), "f"(_fma_70), "f"(r0_23));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r1_23), "f"(_fma_72), "f"(r1_23));
        float2 s_22 = make_float2(_fma_69, _fma_71);
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&_cvt_f32_34), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), _cvt_f32_36);
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_cvt_f32_35));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&_cvt_f32_36));
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y));
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y));
        dgate_b[3] = __as_u32(_bf16x2_22);
        dup_b[3] = __as_u32(_bf16x2_23);
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_25 = approx_exp2((-_cvt_f32_37.x) * 1.442695f);
        float e1_25 = approx_exp2((-_cvt_f32_37.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r0_25), "f"(_fma_75), "f"(r0_25));
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(r1_25), "f"(_fma_77), "f"(r1_25));
        float2 s_24 = make_float2(_fma_74, _fma_76);
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), _cvt_f32_39);
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_cvt_f32_38));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&_cvt_f32_39));
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y));
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y));
        dgate_b[4] = __as_u32(_bf16x2_24);
        dup_b[4] = __as_u32(_bf16x2_25);
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_27 = approx_exp2((-_cvt_f32_40.x) * 1.442695f);
        float e1_27 = approx_exp2((-_cvt_f32_40.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(r0_27), "f"(_fma_80), "f"(r0_27));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r1_27), "f"(_fma_82), "f"(r1_27));
        float2 s_26 = make_float2(_fma_79, _fma_81);
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&_cvt_f32_40), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), _cvt_f32_42);
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_cvt_f32_41));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&_cvt_f32_42));
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y));
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y));
        dgate_b[5] = __as_u32(_bf16x2_26);
        dup_b[5] = __as_u32(_bf16x2_27);
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_29 = approx_exp2((-_cvt_f32_43.x) * 1.442695f);
        float e1_29 = approx_exp2((-_cvt_f32_43.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(r0_29), "f"(_fma_85), "f"(r0_29));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(r1_29), "f"(_fma_87), "f"(r1_29));
        float2 s_28 = make_float2(_fma_84, _fma_86);
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), _cvt_f32_45);
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_cvt_f32_44));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&_cvt_f32_45));
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y));
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y));
        dgate_b[6] = __as_u32(_bf16x2_28);
        dup_b[6] = __as_u32(_bf16x2_29);
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_31 = approx_exp2((-_cvt_f32_46.x) * 1.442695f);
        float e1_31 = approx_exp2((-_cvt_f32_46.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r0_31), "f"(_fma_90), "f"(r0_31));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r1_31), "f"(_fma_92), "f"(r1_31));
        float2 s_30 = make_float2(_fma_89, _fma_91);
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&_cvt_f32_46), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), _cvt_f32_48);
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_cvt_f32_47));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&_cvt_f32_48));
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y));
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y));
        dgate_b[7] = __as_u32(_bf16x2_30);
        dup_b[7] = __as_u32(_bf16x2_31);
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
        float e0_33 = approx_exp2((-_cvt_f32_49.x) * 1.442695f);
        float e1_33 = approx_exp2((-_cvt_f32_49.y) * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(r0_33), "f"(_fma_95), "f"(r0_33));
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(r1_33), "f"(_fma_97), "f"(r1_33));
        float2 s_32 = make_float2(_fma_94, _fma_96);
        float2 act_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), _cvt_f32_51);
        float2 _mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_cvt_f32_50));
        float2 _mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&_cvt_f32_51));
        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y));
        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y));
        dgate_c[0] = __as_u32(_bf16x2_32);
        dup_c[0] = __as_u32(_bf16x2_33);
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[1]));
        float e0_35 = approx_exp2((-_cvt_f32_52.x) * 1.442695f);
        float e1_35 = approx_exp2((-_cvt_f32_52.y) * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(r0_35), "f"(_fma_100), "f"(r0_35));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(r1_35), "f"(_fma_102), "f"(r1_35));
        float2 s_34 = make_float2(_fma_99, _fma_101);
        float2 act_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_34) : "l"(*(const unsigned long long*)&_cvt_f32_52), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_34, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), _cvt_f32_54);
        float2 _mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_cvt_f32_53));
        float2 _mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_35) : "l"(*(const unsigned long long*)&act_34), "l"(*(const unsigned long long*)&_cvt_f32_54));
        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_mul_34.x, _mul_34.y));
        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_mul_35.x, _mul_35.y));
        dgate_c[1] = __as_u32(_bf16x2_34);
        dup_c[1] = __as_u32(_bf16x2_35);
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[2]));
        float e0_37 = approx_exp2((-_cvt_f32_55.x) * 1.442695f);
        float e1_37 = approx_exp2((-_cvt_f32_55.y) * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(r0_37), "f"(_fma_105), "f"(r0_37));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(r1_37), "f"(_fma_107), "f"(r1_37));
        float2 s_36 = make_float2(_fma_104, _fma_106);
        float2 act_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_36) : "l"(*(const unsigned long long*)&_cvt_f32_55), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_108 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_36, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), _cvt_f32_57);
        float2 _mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_cvt_f32_56));
        float2 _mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_37) : "l"(*(const unsigned long long*)&act_36), "l"(*(const unsigned long long*)&_cvt_f32_57));
        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_mul_36.x, _mul_36.y));
        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_mul_37.x, _mul_37.y));
        dgate_c[2] = __as_u32(_bf16x2_36);
        dup_c[2] = __as_u32(_bf16x2_37);
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[3]));
        float e0_39 = approx_exp2((-_cvt_f32_58.x) * 1.442695f);
        float e1_39 = approx_exp2((-_cvt_f32_58.y) * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(r0_39), "f"(_fma_110), "f"(r0_39));
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(r1_39), "f"(_fma_112), "f"(r1_39));
        float2 s_38 = make_float2(_fma_109, _fma_111);
        float2 act_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_38) : "l"(*(const unsigned long long*)&_cvt_f32_58), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_38, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), _cvt_f32_60);
        float2 _mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_cvt_f32_59));
        float2 _mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_39) : "l"(*(const unsigned long long*)&act_38), "l"(*(const unsigned long long*)&_cvt_f32_60));
        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_mul_38.x, _mul_38.y));
        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_mul_39.x, _mul_39.y));
        dgate_c[3] = __as_u32(_bf16x2_38);
        dup_c[3] = __as_u32(_bf16x2_39);
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[4]));
        float e0_41 = approx_exp2((-_cvt_f32_61.x) * 1.442695f);
        float e1_41 = approx_exp2((-_cvt_f32_61.y) * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float _fma_114;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(r0_41), "f"(_fma_115), "f"(r0_41));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(r1_41), "f"(_fma_117), "f"(r1_41));
        float2 s_40 = make_float2(_fma_114, _fma_116);
        float2 act_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_40) : "l"(*(const unsigned long long*)&_cvt_f32_61), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_118 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_40, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), _cvt_f32_63);
        float2 _mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&_cvt_f32_62));
        float2 _mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_41) : "l"(*(const unsigned long long*)&act_40), "l"(*(const unsigned long long*)&_cvt_f32_63));
        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_mul_40.x, _mul_40.y));
        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_mul_41.x, _mul_41.y));
        dgate_c[4] = __as_u32(_bf16x2_40);
        dup_c[4] = __as_u32(_bf16x2_41);
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[5]));
        float e0_43 = approx_exp2((-_cvt_f32_64.x) * 1.442695f);
        float e1_43 = approx_exp2((-_cvt_f32_64.y) * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(r0_43), "f"(_fma_120), "f"(r0_43));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(r1_43), "f"(_fma_122), "f"(r1_43));
        float2 s_42 = make_float2(_fma_119, _fma_121);
        float2 act_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_42) : "l"(*(const unsigned long long*)&_cvt_f32_64), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_42, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), _cvt_f32_66);
        float2 _mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_cvt_f32_65));
        float2 _mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_43) : "l"(*(const unsigned long long*)&act_42), "l"(*(const unsigned long long*)&_cvt_f32_66));
        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_mul_42.x, _mul_42.y));
        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_mul_43.x, _mul_43.y));
        dgate_c[5] = __as_u32(_bf16x2_42);
        dup_c[5] = __as_u32(_bf16x2_43);
        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[6]));
        float e0_45 = approx_exp2((-_cvt_f32_67.x) * 1.442695f);
        float e1_45 = approx_exp2((-_cvt_f32_67.y) * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(r0_45), "f"(_fma_125), "f"(r0_45));
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(r1_45), "f"(_fma_127), "f"(r1_45));
        float2 s_44 = make_float2(_fma_124, _fma_126);
        float2 act_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_44) : "l"(*(const unsigned long long*)&_cvt_f32_67), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_128 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_44, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), _cvt_f32_69);
        float2 _mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_cvt_f32_68));
        float2 _mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_45) : "l"(*(const unsigned long long*)&act_44), "l"(*(const unsigned long long*)&_cvt_f32_69));
        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_mul_44.x, _mul_44.y));
        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_mul_45.x, _mul_45.y));
        dgate_c[6] = __as_u32(_bf16x2_44);
        dup_c[6] = __as_u32(_bf16x2_45);
        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[7]));
        float e0_47 = approx_exp2((-_cvt_f32_70.x) * 1.442695f);
        float e1_47 = approx_exp2((-_cvt_f32_70.y) * 1.442695f);
        float d0_47 = 1.0f + e0_47;
        float d1_47 = 1.0f + e1_47;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
        float _fma_129;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(r0_47), "f"(_fma_130), "f"(r0_47));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(r1_47), "f"(_fma_132), "f"(r1_47));
        float2 s_46 = make_float2(_fma_129, _fma_131);
        float2 act_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_46) : "l"(*(const unsigned long long*)&_cvt_f32_70), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_46, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), _cvt_f32_72);
        float2 _mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&_cvt_f32_71));
        float2 _mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_47) : "l"(*(const unsigned long long*)&act_46), "l"(*(const unsigned long long*)&_cvt_f32_72));
        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_mul_46.x, _mul_46.y));
        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_mul_47.x, _mul_47.y));
        dgate_c[7] = __as_u32(_bf16x2_46);
        dup_c[7] = __as_u32(_bf16x2_47);
        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
        float e0_49 = approx_exp2((-_cvt_f32_73.x) * 1.442695f);
        float e1_49 = approx_exp2((-_cvt_f32_73.y) * 1.442695f);
        float d0_49 = 1.0f + e0_49;
        float d1_49 = 1.0f + e1_49;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(r0_49), "f"(_fma_135), "f"(r0_49));
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(r1_49), "f"(_fma_137), "f"(r1_49));
        float2 s_48 = make_float2(_fma_134, _fma_136);
        float2 act_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_48) : "l"(*(const unsigned long long*)&_cvt_f32_73), "l"(*(const unsigned long long*)&s_48));
        float2 _f32x2_mul_a_138 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_48, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), _cvt_f32_75);
        float2 _mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_cvt_f32_74));
        float2 _mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_49) : "l"(*(const unsigned long long*)&act_48), "l"(*(const unsigned long long*)&_cvt_f32_75));
        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(_mul_48.x, _mul_48.y));
        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(_mul_49.x, _mul_49.y));
        dgate_d[0] = __as_u32(_bf16x2_48);
        dup_d[0] = __as_u32(_bf16x2_49);
        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[1]));
        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[1]));
        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[1]));
        float e0_51 = approx_exp2((-_cvt_f32_76.x) * 1.442695f);
        float e1_51 = approx_exp2((-_cvt_f32_76.y) * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float _fma_139;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(r0_51), "f"(_fma_140), "f"(r0_51));
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(r1_51), "f"(_fma_142), "f"(r1_51));
        float2 s_50 = make_float2(_fma_139, _fma_141);
        float2 act_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_50) : "l"(*(const unsigned long long*)&_cvt_f32_76), "l"(*(const unsigned long long*)&s_50));
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_50, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), _cvt_f32_78);
        float2 _mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&_cvt_f32_77));
        float2 _mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_51) : "l"(*(const unsigned long long*)&act_50), "l"(*(const unsigned long long*)&_cvt_f32_78));
        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(_mul_50.x, _mul_50.y));
        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(_mul_51.x, _mul_51.y));
        dgate_d[1] = __as_u32(_bf16x2_50);
        dup_d[1] = __as_u32(_bf16x2_51);
        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[2]));
        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[2]));
        float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[2]));
        float e0_53 = approx_exp2((-_cvt_f32_79.x) * 1.442695f);
        float e1_53 = approx_exp2((-_cvt_f32_79.y) * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float _fma_144;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(r0_53), "f"(_fma_145), "f"(r0_53));
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(r1_53), "f"(_fma_147), "f"(r1_53));
        float2 s_52 = make_float2(_fma_144, _fma_146);
        float2 act_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_52) : "l"(*(const unsigned long long*)&_cvt_f32_79), "l"(*(const unsigned long long*)&s_52));
        float2 _f32x2_mul_a_148 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_52, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_52), s_52), _cvt_f32_81);
        float2 _mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&_cvt_f32_80));
        float2 _mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_53) : "l"(*(const unsigned long long*)&act_52), "l"(*(const unsigned long long*)&_cvt_f32_81));
        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(_mul_52.x, _mul_52.y));
        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(_mul_53.x, _mul_53.y));
        dgate_d[2] = __as_u32(_bf16x2_52);
        dup_d[2] = __as_u32(_bf16x2_53);
        float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[3]));
        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[3]));
        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[3]));
        float e0_55 = approx_exp2((-_cvt_f32_82.x) * 1.442695f);
        float e1_55 = approx_exp2((-_cvt_f32_82.y) * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float _fma_149;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(r0_55), "f"(_fma_150), "f"(r0_55));
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(r1_55), "f"(_fma_152), "f"(r1_55));
        float2 s_54 = make_float2(_fma_149, _fma_151);
        float2 act_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_54) : "l"(*(const unsigned long long*)&_cvt_f32_82), "l"(*(const unsigned long long*)&s_54));
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_54, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_54), s_54), _cvt_f32_84);
        float2 _mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_cvt_f32_83));
        float2 _mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_55) : "l"(*(const unsigned long long*)&act_54), "l"(*(const unsigned long long*)&_cvt_f32_84));
        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(_mul_54.x, _mul_54.y));
        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(_mul_55.x, _mul_55.y));
        dgate_d[3] = __as_u32(_bf16x2_54);
        dup_d[3] = __as_u32(_bf16x2_55);
        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[4]));
        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[4]));
        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[4]));
        float e0_57 = approx_exp2((-_cvt_f32_85.x) * 1.442695f);
        float e1_57 = approx_exp2((-_cvt_f32_85.y) * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float _fma_154;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(r0_57), "f"(_fma_155), "f"(r0_57));
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(r1_57), "f"(_fma_157), "f"(r1_57));
        float2 s_56 = make_float2(_fma_154, _fma_156);
        float2 act_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_56) : "l"(*(const unsigned long long*)&_cvt_f32_85), "l"(*(const unsigned long long*)&s_56));
        float2 _f32x2_mul_a_158 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_56, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_56), s_56), _cvt_f32_87);
        float2 _mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&_cvt_f32_86));
        float2 _mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_57) : "l"(*(const unsigned long long*)&act_56), "l"(*(const unsigned long long*)&_cvt_f32_87));
        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(_mul_56.x, _mul_56.y));
        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(_mul_57.x, _mul_57.y));
        dgate_d[4] = __as_u32(_bf16x2_56);
        dup_d[4] = __as_u32(_bf16x2_57);
        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[5]));
        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[5]));
        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[5]));
        float e0_59 = approx_exp2((-_cvt_f32_88.x) * 1.442695f);
        float e1_59 = approx_exp2((-_cvt_f32_88.y) * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float _fma_159;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(r0_59), "f"(_fma_160), "f"(r0_59));
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(r1_59), "f"(_fma_162), "f"(r1_59));
        float2 s_58 = make_float2(_fma_159, _fma_161);
        float2 act_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_58) : "l"(*(const unsigned long long*)&_cvt_f32_88), "l"(*(const unsigned long long*)&s_58));
        float2 _f32x2_mul_a_163 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_58, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_58), s_58), _cvt_f32_90);
        float2 _mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&_cvt_f32_89));
        float2 _mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_59) : "l"(*(const unsigned long long*)&act_58), "l"(*(const unsigned long long*)&_cvt_f32_90));
        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(_mul_58.x, _mul_58.y));
        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(_mul_59.x, _mul_59.y));
        dgate_d[5] = __as_u32(_bf16x2_58);
        dup_d[5] = __as_u32(_bf16x2_59);
        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[6]));
        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[6]));
        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[6]));
        float e0_61 = approx_exp2((-_cvt_f32_91.x) * 1.442695f);
        float e1_61 = approx_exp2((-_cvt_f32_91.y) * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float _fma_164;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(r0_61), "f"(_fma_165), "f"(r0_61));
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(r1_61), "f"(_fma_167), "f"(r1_61));
        float2 s_60 = make_float2(_fma_164, _fma_166);
        float2 act_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_60) : "l"(*(const unsigned long long*)&_cvt_f32_91), "l"(*(const unsigned long long*)&s_60));
        float2 _f32x2_mul_a_168 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_60, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_60), s_60), _cvt_f32_93);
        float2 _mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&_cvt_f32_92));
        float2 _mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_61) : "l"(*(const unsigned long long*)&act_60), "l"(*(const unsigned long long*)&_cvt_f32_93));
        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(_mul_60.x, _mul_60.y));
        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(_mul_61.x, _mul_61.y));
        dgate_d[6] = __as_u32(_bf16x2_60);
        dup_d[6] = __as_u32(_bf16x2_61);
        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[7]));
        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[7]));
        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[7]));
        float e0_63 = approx_exp2((-_cvt_f32_94.x) * 1.442695f);
        float e1_63 = approx_exp2((-_cvt_f32_94.y) * 1.442695f);
        float d0_63 = 1.0f + e0_63;
        float d1_63 = 1.0f + e1_63;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_63), "f"(r0_63), "f"(1.0f));
        float _fma_169;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(r0_63), "f"(_fma_170), "f"(r0_63));
        float _fma_172;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(-d1_63), "f"(r1_63), "f"(1.0f));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(r1_63), "f"(_fma_172), "f"(r1_63));
        float2 s_62 = make_float2(_fma_169, _fma_171);
        float2 act_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_62) : "l"(*(const unsigned long long*)&_cvt_f32_94), "l"(*(const unsigned long long*)&s_62));
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_62, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_62), s_62), _cvt_f32_96);
        float2 _mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&_cvt_f32_95));
        float2 _mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_63) : "l"(*(const unsigned long long*)&act_62), "l"(*(const unsigned long long*)&_cvt_f32_96));
        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(_mul_62.x, _mul_62.y));
        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(_mul_63.x, _mul_63.y));
        dgate_d[7] = __as_u32(_bf16x2_62);
        dup_d[7] = __as_u32(_bf16x2_63);
    } else {
        int band = 0;
        float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_65 = approx_exp2((-_cvt_f32_97.x) * 1.442695f);
        float e1_65 = approx_exp2((-_cvt_f32_97.y) * 1.442695f);
        float d0_65 = 1.0f + e0_65;
        float d1_65 = 1.0f + e1_65;
        float r0_65 = approx_rcp(d0_65);
        float r1_65 = approx_rcp(d1_65);
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d0_65), "f"(r0_65), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_65), "f"(_fma_174), "f"(r0_65));
        float _fma_175;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(-d1_65), "f"(r1_65), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_65), "f"(_fma_175), "f"(r1_65));
        float t0_65 = _fma_0;
        float t1_65 = _fma_1;
        band = band | ((fminf(t0_65, t1_65) == 0.0f) ? 1 : 0);
        float2 s_64 = make_float2(max_noftz(t0_65, 0.0f), max_noftz(t1_65, 0.0f));
        float2 act_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_64) : "l"(*(const unsigned long long*)&_cvt_f32_97), "l"(*(const unsigned long long*)&s_64));
        float2 _f32x2_mul_a_176 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_64, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_64), s_64), _cvt_f32_99);
        float2 _mul_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_cvt_f32_98));
        float2 _mul_65;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_65) : "l"(*(const unsigned long long*)&act_64), "l"(*(const unsigned long long*)&_cvt_f32_99));
        __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(_mul_64.x, _mul_64.y));
        __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(_mul_65.x, _mul_65.y));
        dgate[0] = __as_u32(_bf16x2_64);
        dup[0] = __as_u32(_bf16x2_65);
        float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_67 = approx_exp2((-_cvt_f32_100.x) * 1.442695f);
        float e1_67 = approx_exp2((-_cvt_f32_100.y) * 1.442695f);
        float d0_67 = 1.0f + e0_67;
        float d1_67 = 1.0f + e1_67;
        float r0_67 = approx_rcp(d0_67);
        float r1_67 = approx_rcp(d1_67);
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d0_67), "f"(r0_67), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_67), "f"(_fma_177), "f"(r0_67));
        float _fma_178;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(-d1_67), "f"(r1_67), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_67), "f"(_fma_178), "f"(r1_67));
        float t0_67 = _fma_2;
        float t1_67 = _fma_3;
        band = band | ((fminf(t0_67, t1_67) == 0.0f) ? 2 : 0);
        float2 s_66 = make_float2(max_noftz(t0_67, 0.0f), max_noftz(t1_67, 0.0f));
        float2 act_66;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_66) : "l"(*(const unsigned long long*)&_cvt_f32_100), "l"(*(const unsigned long long*)&s_66));
        float2 _f32x2_mul_a_179 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_66, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_66), s_66), _cvt_f32_102);
        float2 _mul_66;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&_cvt_f32_101));
        float2 _mul_67;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_67) : "l"(*(const unsigned long long*)&act_66), "l"(*(const unsigned long long*)&_cvt_f32_102));
        __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(_mul_66.x, _mul_66.y));
        __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(_mul_67.x, _mul_67.y));
        dgate[1] = __as_u32(_bf16x2_66);
        dup[1] = __as_u32(_bf16x2_67);
        float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_69 = approx_exp2((-_cvt_f32_103.x) * 1.442695f);
        float e1_69 = approx_exp2((-_cvt_f32_103.y) * 1.442695f);
        float d0_69 = 1.0f + e0_69;
        float d1_69 = 1.0f + e1_69;
        float r0_69 = approx_rcp(d0_69);
        float r1_69 = approx_rcp(d1_69);
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d0_69), "f"(r0_69), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_69), "f"(_fma_180), "f"(r0_69));
        float _fma_181;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(-d1_69), "f"(r1_69), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r1_69), "f"(_fma_181), "f"(r1_69));
        float t0_69 = _fma_4;
        float t1_69 = _fma_5;
        band = band | ((fminf(t0_69, t1_69) == 0.0f) ? 4 : 0);
        float2 s_68 = make_float2(max_noftz(t0_69, 0.0f), max_noftz(t1_69, 0.0f));
        float2 act_68;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_68) : "l"(*(const unsigned long long*)&_cvt_f32_103), "l"(*(const unsigned long long*)&s_68));
        float2 _f32x2_mul_a_182 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_68, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_68), s_68), _cvt_f32_105);
        float2 _mul_68;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_68) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&_cvt_f32_104));
        float2 _mul_69;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_69) : "l"(*(const unsigned long long*)&act_68), "l"(*(const unsigned long long*)&_cvt_f32_105));
        __nv_bfloat162 _bf16x2_68 = __float22bfloat162_rn(make_float2(_mul_68.x, _mul_68.y));
        __nv_bfloat162 _bf16x2_69 = __float22bfloat162_rn(make_float2(_mul_69.x, _mul_69.y));
        dgate[2] = __as_u32(_bf16x2_68);
        dup[2] = __as_u32(_bf16x2_69);
        float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_71 = approx_exp2((-_cvt_f32_106.x) * 1.442695f);
        float e1_71 = approx_exp2((-_cvt_f32_106.y) * 1.442695f);
        float d0_71 = 1.0f + e0_71;
        float d1_71 = 1.0f + e1_71;
        float r0_71 = approx_rcp(d0_71);
        float r1_71 = approx_rcp(d1_71);
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(-d0_71), "f"(r0_71), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r0_71), "f"(_fma_183), "f"(r0_71));
        float _fma_184;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(-d1_71), "f"(r1_71), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1_71), "f"(_fma_184), "f"(r1_71));
        float t0_71 = _fma_6;
        float t1_71 = _fma_7;
        band = band | ((fminf(t0_71, t1_71) == 0.0f) ? 8 : 0);
        float2 s_70 = make_float2(max_noftz(t0_71, 0.0f), max_noftz(t1_71, 0.0f));
        float2 act_70;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_70) : "l"(*(const unsigned long long*)&_cvt_f32_106), "l"(*(const unsigned long long*)&s_70));
        float2 _f32x2_mul_a_185 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_70, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_70), s_70), _cvt_f32_108);
        float2 _mul_70;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_70) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&_cvt_f32_107));
        float2 _mul_71;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_71) : "l"(*(const unsigned long long*)&act_70), "l"(*(const unsigned long long*)&_cvt_f32_108));
        __nv_bfloat162 _bf16x2_70 = __float22bfloat162_rn(make_float2(_mul_70.x, _mul_70.y));
        __nv_bfloat162 _bf16x2_71 = __float22bfloat162_rn(make_float2(_mul_71.x, _mul_71.y));
        dgate[3] = __as_u32(_bf16x2_70);
        dup[3] = __as_u32(_bf16x2_71);
        float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_73 = approx_exp2((-_cvt_f32_109.x) * 1.442695f);
        float e1_73 = approx_exp2((-_cvt_f32_109.y) * 1.442695f);
        float d0_73 = 1.0f + e0_73;
        float d1_73 = 1.0f + e1_73;
        float r0_73 = approx_rcp(d0_73);
        float r1_73 = approx_rcp(d1_73);
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(-d0_73), "f"(r0_73), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r0_73), "f"(_fma_186), "f"(r0_73));
        float _fma_187;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(-d1_73), "f"(r1_73), "f"(1.0f));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r1_73), "f"(_fma_187), "f"(r1_73));
        float t0_73 = _fma_8;
        float t1_73 = _fma_9;
        band = band | ((fminf(t0_73, t1_73) == 0.0f) ? 16 : 0);
        float2 s_72 = make_float2(max_noftz(t0_73, 0.0f), max_noftz(t1_73, 0.0f));
        float2 act_72;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_72) : "l"(*(const unsigned long long*)&_cvt_f32_109), "l"(*(const unsigned long long*)&s_72));
        float2 _f32x2_mul_a_188 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_72, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_72), s_72), _cvt_f32_111);
        float2 _mul_72;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&_cvt_f32_110));
        float2 _mul_73;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_73) : "l"(*(const unsigned long long*)&act_72), "l"(*(const unsigned long long*)&_cvt_f32_111));
        __nv_bfloat162 _bf16x2_72 = __float22bfloat162_rn(make_float2(_mul_72.x, _mul_72.y));
        __nv_bfloat162 _bf16x2_73 = __float22bfloat162_rn(make_float2(_mul_73.x, _mul_73.y));
        dgate[4] = __as_u32(_bf16x2_72);
        dup[4] = __as_u32(_bf16x2_73);
        float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_75 = approx_exp2((-_cvt_f32_112.x) * 1.442695f);
        float e1_75 = approx_exp2((-_cvt_f32_112.y) * 1.442695f);
        float d0_75 = 1.0f + e0_75;
        float d1_75 = 1.0f + e1_75;
        float r0_75 = approx_rcp(d0_75);
        float r1_75 = approx_rcp(d1_75);
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d0_75), "f"(r0_75), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_75), "f"(_fma_189), "f"(r0_75));
        float _fma_190;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(-d1_75), "f"(r1_75), "f"(1.0f));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(r1_75), "f"(_fma_190), "f"(r1_75));
        float t0_75 = _fma_10;
        float t1_75 = _fma_11;
        band = band | ((fminf(t0_75, t1_75) == 0.0f) ? 32 : 0);
        float2 s_74 = make_float2(max_noftz(t0_75, 0.0f), max_noftz(t1_75, 0.0f));
        float2 act_74;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_74) : "l"(*(const unsigned long long*)&_cvt_f32_112), "l"(*(const unsigned long long*)&s_74));
        float2 _f32x2_mul_a_191 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_74, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_74), s_74), _cvt_f32_114);
        float2 _mul_74;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_74) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&_cvt_f32_113));
        float2 _mul_75;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_75) : "l"(*(const unsigned long long*)&act_74), "l"(*(const unsigned long long*)&_cvt_f32_114));
        __nv_bfloat162 _bf16x2_74 = __float22bfloat162_rn(make_float2(_mul_74.x, _mul_74.y));
        __nv_bfloat162 _bf16x2_75 = __float22bfloat162_rn(make_float2(_mul_75.x, _mul_75.y));
        dgate[5] = __as_u32(_bf16x2_74);
        dup[5] = __as_u32(_bf16x2_75);
        float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_117 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_77 = approx_exp2((-_cvt_f32_115.x) * 1.442695f);
        float e1_77 = approx_exp2((-_cvt_f32_115.y) * 1.442695f);
        float d0_77 = 1.0f + e0_77;
        float d1_77 = 1.0f + e1_77;
        float r0_77 = approx_rcp(d0_77);
        float r1_77 = approx_rcp(d1_77);
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(-d0_77), "f"(r0_77), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r0_77), "f"(_fma_192), "f"(r0_77));
        float _fma_193;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(-d1_77), "f"(r1_77), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r1_77), "f"(_fma_193), "f"(r1_77));
        float t0_77 = _fma_12;
        float t1_77 = _fma_13;
        band = band | ((fminf(t0_77, t1_77) == 0.0f) ? 64 : 0);
        float2 s_76 = make_float2(max_noftz(t0_77, 0.0f), max_noftz(t1_77, 0.0f));
        float2 act_76;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_76) : "l"(*(const unsigned long long*)&_cvt_f32_115), "l"(*(const unsigned long long*)&s_76));
        float2 _f32x2_mul_a_194 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_76, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_76), s_76), _cvt_f32_117);
        float2 _mul_76;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&_cvt_f32_116));
        float2 _mul_77;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_77) : "l"(*(const unsigned long long*)&act_76), "l"(*(const unsigned long long*)&_cvt_f32_117));
        __nv_bfloat162 _bf16x2_76 = __float22bfloat162_rn(make_float2(_mul_76.x, _mul_76.y));
        __nv_bfloat162 _bf16x2_77 = __float22bfloat162_rn(make_float2(_mul_77.x, _mul_77.y));
        dgate[6] = __as_u32(_bf16x2_76);
        dup[6] = __as_u32(_bf16x2_77);
        float2 _cvt_f32_118 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_119 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_120 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_79 = approx_exp2((-_cvt_f32_118.x) * 1.442695f);
        float e1_79 = approx_exp2((-_cvt_f32_118.y) * 1.442695f);
        float d0_79 = 1.0f + e0_79;
        float d1_79 = 1.0f + e1_79;
        float r0_79 = approx_rcp(d0_79);
        float r1_79 = approx_rcp(d1_79);
        float _fma_195;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d0_79), "f"(r0_79), "f"(1.0f));
        float _fma_14_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14_1) : "f"(r0_79), "f"(_fma_195), "f"(r0_79));
        float _fma_196;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(-d1_79), "f"(r1_79), "f"(1.0f));
        float _fma_15_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15_1) : "f"(r1_79), "f"(_fma_196), "f"(r1_79));
        float t0_79 = _fma_14_1;
        float t1_79 = _fma_15_1;
        band = band | ((fminf(t0_79, t1_79) == 0.0f) ? 128 : 0);
        float2 s_78 = make_float2(max_noftz(t0_79, 0.0f), max_noftz(t1_79, 0.0f));
        float2 act_78;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_78) : "l"(*(const unsigned long long*)&_cvt_f32_118), "l"(*(const unsigned long long*)&s_78));
        float2 _f32x2_mul_a_197 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_78, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_78), s_78), _cvt_f32_120);
        float2 _mul_78;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&_cvt_f32_119));
        float2 _mul_79;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_79) : "l"(*(const unsigned long long*)&act_78), "l"(*(const unsigned long long*)&_cvt_f32_120));
        __nv_bfloat162 _bf16x2_78 = __float22bfloat162_rn(make_float2(_mul_78.x, _mul_78.y));
        __nv_bfloat162 _bf16x2_79 = __float22bfloat162_rn(make_float2(_mul_79.x, _mul_79.y));
        dgate[7] = __as_u32(_bf16x2_78);
        dup[7] = __as_u32(_bf16x2_79);
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_12[1];
                    {
                        _vec_load_12[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_13[1];
                    {
                        _vec_load_13[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    unsigned int _vec_load_14[1];
                    {
                        _vec_load_14[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * i) + 0);
                    }
                    float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_12[0]));
                    float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_13[0]));
                    float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_14[0]));
                    float e0_81 = approx_exp2((-_cvt_f32_121.x) * 1.442695f);
                    float e1_81 = approx_exp2((-_cvt_f32_121.y) * 1.442695f);
                    float d0_81 = 1.0f + e0_81;
                    float d1_81 = 1.0f + e1_81;
                    float r0_81 = approx_rcp(d0_81);
                    float r1_81 = approx_rcp(d1_81);
                    float _fma_198;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(-d0_81), "f"(r0_81), "f"(1.0f));
                    float _fma_16_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_81), "f"(_fma_198), "f"(r0_81));
                    float _fma_199;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(-d1_81), "f"(r1_81), "f"(1.0f));
                    float _fma_17_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17_1) : "f"(r1_81), "f"(_fma_199), "f"(r1_81));
                    float t0_81 = _fma_16_1;
                    float t1_81 = _fma_17_1;
                    float d4_81a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_81a) : "f"(d0_81), "f"(0.25f));
                    float r4_81a = approx_rcp(d4_81a);
                    float _fma_201;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(-d4_81a), "f"(r4_81a), "f"(1.0f));
                    float _fma_200;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(r4_81a), "f"(_fma_201), "f"(r4_81a));
                    float big_81a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_81a) : "f"(mul_f32_rn_noftz(_fma_200, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(big_81a), "f"(8.388608e+06f));
                    float whole_81a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_81a) : "f"(_add_202), "f"(8.388608e+06f));
                    float above_81a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_81a) : "f"(-d4_81a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_81a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_204;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_204) : "f"(big_81a), "f"(whole_81a));
                    float _abs_203;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(_sub_204));
                    float _add_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(big_81a), "f"(0.5f));
                    float _sub_206;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_206) : "f"(big_81a), "f"(0.5f));
                    float rounded_81a = ((_abs_203 == 0.5f) ? ((above_81a > 0.0f) ? _add_205 : _sub_206) : whole_81a);
                    float u0_81 = ((d0_81 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_81a, 7.888609e-31f), 1.7763568e-15f) : t0_81);
                    float d4_81b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_81b) : "f"(d1_81), "f"(0.25f));
                    float r4_81b = approx_rcp(d4_81b);
                    float _fma_208;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(-d4_81b), "f"(r4_81b), "f"(1.0f));
                    float _fma_207;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(r4_81b), "f"(_fma_208), "f"(r4_81b));
                    float big_81b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_81b) : "f"(mul_f32_rn_noftz(_fma_207, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_209;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(big_81b), "f"(8.388608e+06f));
                    float whole_81b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_81b) : "f"(_add_209), "f"(8.388608e+06f));
                    float above_81b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_81b) : "f"(-d4_81b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_81b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_211;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_211) : "f"(big_81b), "f"(whole_81b));
                    float _abs_210;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(_sub_211));
                    float _add_212;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(big_81b), "f"(0.5f));
                    float _sub_213;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(big_81b), "f"(0.5f));
                    float rounded_81b = ((_abs_210 == 0.5f) ? ((above_81b > 0.0f) ? _add_212 : _sub_213) : whole_81b);
                    float u1_81 = ((d1_81 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_81b, 7.888609e-31f), 1.7763568e-15f) : t1_81);
                    float2 s_80 = make_float2(max_noftz(u0_81, 0.0f), max_noftz(u1_81, 0.0f));
                    float2 act_80;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_80) : "l"(*(const unsigned long long*)&_cvt_f32_121), "l"(*(const unsigned long long*)&s_80));
                    float2 _f32x2_mul_a_214 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_80, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_80), s_80), _cvt_f32_123);
                    float2 _mul_80;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_80) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&_cvt_f32_122));
                    float2 _mul_81;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_81) : "l"(*(const unsigned long long*)&act_80), "l"(*(const unsigned long long*)&_cvt_f32_123));
                    __nv_bfloat162 _bf16x2_80 = __float22bfloat162_rn(make_float2(_mul_80.x, _mul_80.y));
                    __nv_bfloat162 _bf16x2_81 = __float22bfloat162_rn(make_float2(_mul_81.x, _mul_81.y));
                    unsigned int dgate_new = __as_u32(_bf16x2_80);
                    unsigned int dup_new = __as_u32(_bf16x2_81);
                    dgate[0] = ((i == 0) ? dgate_new : dgate[0]);
                    dup[0] = ((i == 0) ? dup_new : dup[0]);
                    dgate[1] = ((i == 1) ? dgate_new : dgate[1]);
                    dup[1] = ((i == 1) ? dup_new : dup[1]);
                    dgate[2] = ((i == 2) ? dgate_new : dgate[2]);
                    dup[2] = ((i == 2) ? dup_new : dup[2]);
                    dgate[3] = ((i == 3) ? dgate_new : dgate[3]);
                    dup[3] = ((i == 3) ? dup_new : dup[3]);
                    dgate[4] = ((i == 4) ? dgate_new : dgate[4]);
                    dup[4] = ((i == 4) ? dup_new : dup[4]);
                    dgate[5] = ((i == 5) ? dgate_new : dgate[5]);
                    dup[5] = ((i == 5) ? dup_new : dup[5]);
                    dgate[6] = ((i == 6) ? dgate_new : dgate[6]);
                    dup[6] = ((i == 6) ? dup_new : dup[6]);
                    dgate[7] = ((i == 7) ? dgate_new : dgate[7]);
                    dup[7] = ((i == 7) ? dup_new : dup[7]);
                }
            }
        }
        int band_0 = 0;
        float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_83 = approx_exp2((-_cvt_f32_124.x) * 1.442695f);
        float e1_83 = approx_exp2((-_cvt_f32_124.y) * 1.442695f);
        float d0_83 = 1.0f + e0_83;
        float d1_83 = 1.0f + e1_83;
        float r0_83 = approx_rcp(d0_83);
        float r1_83 = approx_rcp(d1_83);
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d0_83), "f"(r0_83), "f"(1.0f));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(r0_83), "f"(_fma_215), "f"(r0_83));
        float _fma_216;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(-d1_83), "f"(r1_83), "f"(1.0f));
        float _fma_19_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19_1) : "f"(r1_83), "f"(_fma_216), "f"(r1_83));
        float t0_83 = _fma_18;
        float t1_83 = _fma_19_1;
        band_0 = band_0 | ((fminf(t0_83, t1_83) == 0.0f) ? 1 : 0);
        float2 s_82 = make_float2(max_noftz(t0_83, 0.0f), max_noftz(t1_83, 0.0f));
        float2 act_82;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_82) : "l"(*(const unsigned long long*)&_cvt_f32_124), "l"(*(const unsigned long long*)&s_82));
        float2 _f32x2_mul_a_217 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_82, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_82), s_82), _cvt_f32_126);
        float2 _mul_82;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_217), "l"(*(const unsigned long long*)&_cvt_f32_125));
        float2 _mul_83;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_83) : "l"(*(const unsigned long long*)&act_82), "l"(*(const unsigned long long*)&_cvt_f32_126));
        __nv_bfloat162 _bf16x2_82 = __float22bfloat162_rn(make_float2(_mul_82.x, _mul_82.y));
        __nv_bfloat162 _bf16x2_83 = __float22bfloat162_rn(make_float2(_mul_83.x, _mul_83.y));
        dgate_b[0] = __as_u32(_bf16x2_82);
        dup_b[0] = __as_u32(_bf16x2_83);
        float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_128 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_129 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_85 = approx_exp2((-_cvt_f32_127.x) * 1.442695f);
        float e1_85 = approx_exp2((-_cvt_f32_127.y) * 1.442695f);
        float d0_85 = 1.0f + e0_85;
        float d1_85 = 1.0f + e1_85;
        float r0_85 = approx_rcp(d0_85);
        float r1_85 = approx_rcp(d1_85);
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d0_85), "f"(r0_85), "f"(1.0f));
        float _fma_20_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20_1) : "f"(r0_85), "f"(_fma_218), "f"(r0_85));
        float _fma_219;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(-d1_85), "f"(r1_85), "f"(1.0f));
        float _fma_21_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21_1) : "f"(r1_85), "f"(_fma_219), "f"(r1_85));
        float t0_85 = _fma_20_1;
        float t1_85 = _fma_21_1;
        band_0 = band_0 | ((fminf(t0_85, t1_85) == 0.0f) ? 2 : 0);
        float2 s_84 = make_float2(max_noftz(t0_85, 0.0f), max_noftz(t1_85, 0.0f));
        float2 act_84;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_84) : "l"(*(const unsigned long long*)&_cvt_f32_127), "l"(*(const unsigned long long*)&s_84));
        float2 _f32x2_mul_a_220 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_84, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_84), s_84), _cvt_f32_129);
        float2 _mul_84;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&_cvt_f32_128));
        float2 _mul_85;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_85) : "l"(*(const unsigned long long*)&act_84), "l"(*(const unsigned long long*)&_cvt_f32_129));
        __nv_bfloat162 _bf16x2_84 = __float22bfloat162_rn(make_float2(_mul_84.x, _mul_84.y));
        __nv_bfloat162 _bf16x2_85 = __float22bfloat162_rn(make_float2(_mul_85.x, _mul_85.y));
        dgate_b[1] = __as_u32(_bf16x2_84);
        dup_b[1] = __as_u32(_bf16x2_85);
        float2 _cvt_f32_130 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_131 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_132 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_87 = approx_exp2((-_cvt_f32_130.x) * 1.442695f);
        float e1_87 = approx_exp2((-_cvt_f32_130.y) * 1.442695f);
        float d0_87 = 1.0f + e0_87;
        float d1_87 = 1.0f + e1_87;
        float r0_87 = approx_rcp(d0_87);
        float r1_87 = approx_rcp(d1_87);
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(-d0_87), "f"(r0_87), "f"(1.0f));
        float _fma_22_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22_1) : "f"(r0_87), "f"(_fma_221), "f"(r0_87));
        float _fma_222;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(-d1_87), "f"(r1_87), "f"(1.0f));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r1_87), "f"(_fma_222), "f"(r1_87));
        float t0_87 = _fma_22_1;
        float t1_87 = _fma_23;
        band_0 = band_0 | ((fminf(t0_87, t1_87) == 0.0f) ? 4 : 0);
        float2 s_86 = make_float2(max_noftz(t0_87, 0.0f), max_noftz(t1_87, 0.0f));
        float2 act_86;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_86) : "l"(*(const unsigned long long*)&_cvt_f32_130), "l"(*(const unsigned long long*)&s_86));
        float2 _f32x2_mul_a_223 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_86, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_86), s_86), _cvt_f32_132);
        float2 _mul_86;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&_cvt_f32_131));
        float2 _mul_87;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_87) : "l"(*(const unsigned long long*)&act_86), "l"(*(const unsigned long long*)&_cvt_f32_132));
        __nv_bfloat162 _bf16x2_86 = __float22bfloat162_rn(make_float2(_mul_86.x, _mul_86.y));
        __nv_bfloat162 _bf16x2_87 = __float22bfloat162_rn(make_float2(_mul_87.x, _mul_87.y));
        dgate_b[2] = __as_u32(_bf16x2_86);
        dup_b[2] = __as_u32(_bf16x2_87);
        float2 _cvt_f32_133 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_134 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_135 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_89 = approx_exp2((-_cvt_f32_133.x) * 1.442695f);
        float e1_89 = approx_exp2((-_cvt_f32_133.y) * 1.442695f);
        float d0_89 = 1.0f + e0_89;
        float d1_89 = 1.0f + e1_89;
        float r0_89 = approx_rcp(d0_89);
        float r1_89 = approx_rcp(d1_89);
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(-d0_89), "f"(r0_89), "f"(1.0f));
        float _fma_24_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24_1) : "f"(r0_89), "f"(_fma_224), "f"(r0_89));
        float _fma_225;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(-d1_89), "f"(r1_89), "f"(1.0f));
        float _fma_25_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25_1) : "f"(r1_89), "f"(_fma_225), "f"(r1_89));
        float t0_89 = _fma_24_1;
        float t1_89 = _fma_25_1;
        band_0 = band_0 | ((fminf(t0_89, t1_89) == 0.0f) ? 8 : 0);
        float2 s_88 = make_float2(max_noftz(t0_89, 0.0f), max_noftz(t1_89, 0.0f));
        float2 act_88;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_88) : "l"(*(const unsigned long long*)&_cvt_f32_133), "l"(*(const unsigned long long*)&s_88));
        float2 _f32x2_mul_a_226 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_88, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_88), s_88), _cvt_f32_135);
        float2 _mul_88;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&_cvt_f32_134));
        float2 _mul_89;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_89) : "l"(*(const unsigned long long*)&act_88), "l"(*(const unsigned long long*)&_cvt_f32_135));
        __nv_bfloat162 _bf16x2_88 = __float22bfloat162_rn(make_float2(_mul_88.x, _mul_88.y));
        __nv_bfloat162 _bf16x2_89 = __float22bfloat162_rn(make_float2(_mul_89.x, _mul_89.y));
        dgate_b[3] = __as_u32(_bf16x2_88);
        dup_b[3] = __as_u32(_bf16x2_89);
        float2 _cvt_f32_136 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_137 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_138 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_91 = approx_exp2((-_cvt_f32_136.x) * 1.442695f);
        float e1_91 = approx_exp2((-_cvt_f32_136.y) * 1.442695f);
        float d0_91 = 1.0f + e0_91;
        float d1_91 = 1.0f + e1_91;
        float r0_91 = approx_rcp(d0_91);
        float r1_91 = approx_rcp(d1_91);
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(-d0_91), "f"(r0_91), "f"(1.0f));
        float _fma_26_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26_1) : "f"(r0_91), "f"(_fma_227), "f"(r0_91));
        float _fma_228;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(-d1_91), "f"(r1_91), "f"(1.0f));
        float _fma_27_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27_1) : "f"(r1_91), "f"(_fma_228), "f"(r1_91));
        float t0_91 = _fma_26_1;
        float t1_91 = _fma_27_1;
        band_0 = band_0 | ((fminf(t0_91, t1_91) == 0.0f) ? 16 : 0);
        float2 s_90 = make_float2(max_noftz(t0_91, 0.0f), max_noftz(t1_91, 0.0f));
        float2 act_90;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_90) : "l"(*(const unsigned long long*)&_cvt_f32_136), "l"(*(const unsigned long long*)&s_90));
        float2 _f32x2_mul_a_229 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_90, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_90), s_90), _cvt_f32_138);
        float2 _mul_90;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&_cvt_f32_137));
        float2 _mul_91;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_91) : "l"(*(const unsigned long long*)&act_90), "l"(*(const unsigned long long*)&_cvt_f32_138));
        __nv_bfloat162 _bf16x2_90 = __float22bfloat162_rn(make_float2(_mul_90.x, _mul_90.y));
        __nv_bfloat162 _bf16x2_91 = __float22bfloat162_rn(make_float2(_mul_91.x, _mul_91.y));
        dgate_b[4] = __as_u32(_bf16x2_90);
        dup_b[4] = __as_u32(_bf16x2_91);
        float2 _cvt_f32_139 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_140 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_141 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_93 = approx_exp2((-_cvt_f32_139.x) * 1.442695f);
        float e1_93 = approx_exp2((-_cvt_f32_139.y) * 1.442695f);
        float d0_93 = 1.0f + e0_93;
        float d1_93 = 1.0f + e1_93;
        float r0_93 = approx_rcp(d0_93);
        float r1_93 = approx_rcp(d1_93);
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d0_93), "f"(r0_93), "f"(1.0f));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r0_93), "f"(_fma_230), "f"(r0_93));
        float _fma_231;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(-d1_93), "f"(r1_93), "f"(1.0f));
        float _fma_29_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29_1) : "f"(r1_93), "f"(_fma_231), "f"(r1_93));
        float t0_93 = _fma_28;
        float t1_93 = _fma_29_1;
        band_0 = band_0 | ((fminf(t0_93, t1_93) == 0.0f) ? 32 : 0);
        float2 s_92 = make_float2(max_noftz(t0_93, 0.0f), max_noftz(t1_93, 0.0f));
        float2 act_92;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_92) : "l"(*(const unsigned long long*)&_cvt_f32_139), "l"(*(const unsigned long long*)&s_92));
        float2 _f32x2_mul_a_232 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_92, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_92), s_92), _cvt_f32_141);
        float2 _mul_92;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&_cvt_f32_140));
        float2 _mul_93;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_93) : "l"(*(const unsigned long long*)&act_92), "l"(*(const unsigned long long*)&_cvt_f32_141));
        __nv_bfloat162 _bf16x2_92 = __float22bfloat162_rn(make_float2(_mul_92.x, _mul_92.y));
        __nv_bfloat162 _bf16x2_93 = __float22bfloat162_rn(make_float2(_mul_93.x, _mul_93.y));
        dgate_b[5] = __as_u32(_bf16x2_92);
        dup_b[5] = __as_u32(_bf16x2_93);
        float2 _cvt_f32_142 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_143 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_144 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_95 = approx_exp2((-_cvt_f32_142.x) * 1.442695f);
        float e1_95 = approx_exp2((-_cvt_f32_142.y) * 1.442695f);
        float d0_95 = 1.0f + e0_95;
        float d1_95 = 1.0f + e1_95;
        float r0_95 = approx_rcp(d0_95);
        float r1_95 = approx_rcp(d1_95);
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d0_95), "f"(r0_95), "f"(1.0f));
        float _fma_30_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30_1) : "f"(r0_95), "f"(_fma_233), "f"(r0_95));
        float _fma_234;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(-d1_95), "f"(r1_95), "f"(1.0f));
        float _fma_31_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31_1) : "f"(r1_95), "f"(_fma_234), "f"(r1_95));
        float t0_95 = _fma_30_1;
        float t1_95 = _fma_31_1;
        band_0 = band_0 | ((fminf(t0_95, t1_95) == 0.0f) ? 64 : 0);
        float2 s_94 = make_float2(max_noftz(t0_95, 0.0f), max_noftz(t1_95, 0.0f));
        float2 act_94;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_94) : "l"(*(const unsigned long long*)&_cvt_f32_142), "l"(*(const unsigned long long*)&s_94));
        float2 _f32x2_mul_a_235 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_94, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_94), s_94), _cvt_f32_144);
        float2 _mul_94;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&_cvt_f32_143));
        float2 _mul_95;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_95) : "l"(*(const unsigned long long*)&act_94), "l"(*(const unsigned long long*)&_cvt_f32_144));
        __nv_bfloat162 _bf16x2_94 = __float22bfloat162_rn(make_float2(_mul_94.x, _mul_94.y));
        __nv_bfloat162 _bf16x2_95 = __float22bfloat162_rn(make_float2(_mul_95.x, _mul_95.y));
        dgate_b[6] = __as_u32(_bf16x2_94);
        dup_b[6] = __as_u32(_bf16x2_95);
        float2 _cvt_f32_145 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_146 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_147 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_97 = approx_exp2((-_cvt_f32_145.x) * 1.442695f);
        float e1_97 = approx_exp2((-_cvt_f32_145.y) * 1.442695f);
        float d0_97 = 1.0f + e0_97;
        float d1_97 = 1.0f + e1_97;
        float r0_97 = approx_rcp(d0_97);
        float r1_97 = approx_rcp(d1_97);
        float _fma_236;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d0_97), "f"(r0_97), "f"(1.0f));
        float _fma_32_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32_1) : "f"(r0_97), "f"(_fma_236), "f"(r0_97));
        float _fma_237;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(-d1_97), "f"(r1_97), "f"(1.0f));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(r1_97), "f"(_fma_237), "f"(r1_97));
        float t0_97 = _fma_32_1;
        float t1_97 = _fma_33;
        band_0 = band_0 | ((fminf(t0_97, t1_97) == 0.0f) ? 128 : 0);
        float2 s_96 = make_float2(max_noftz(t0_97, 0.0f), max_noftz(t1_97, 0.0f));
        float2 act_96;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_96) : "l"(*(const unsigned long long*)&_cvt_f32_145), "l"(*(const unsigned long long*)&s_96));
        float2 _f32x2_mul_a_238 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_96, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_96), s_96), _cvt_f32_147);
        float2 _mul_96;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&_cvt_f32_146));
        float2 _mul_97;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_97) : "l"(*(const unsigned long long*)&act_96), "l"(*(const unsigned long long*)&_cvt_f32_147));
        __nv_bfloat162 _bf16x2_96 = __float22bfloat162_rn(make_float2(_mul_96.x, _mul_96.y));
        __nv_bfloat162 _bf16x2_97 = __float22bfloat162_rn(make_float2(_mul_97.x, _mul_97.y));
        dgate_b[7] = __as_u32(_bf16x2_96);
        dup_b[7] = __as_u32(_bf16x2_97);
        if (band_0 != 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 8; i_1++) {
                if ((band_0 >> i_1 & 1) != 0) {
                    unsigned int _vec_load_15[1];
                    {
                        _vec_load_15[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_0 + 2 * i_1) + 0);
                    }
                    unsigned int _vec_load_16[1];
                    {
                        _vec_load_16[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_0 + K + 2 * i_1) + 0);
                    }
                    unsigned int _vec_load_17[1];
                    {
                        _vec_load_17[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_b * K + col + 2 * i_1) + 0);
                    }
                    float2 _cvt_f32_148 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_15[0]));
                    float2 _cvt_f32_149 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_16[0]));
                    float2 _cvt_f32_150 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_17[0]));
                    float e0_99 = approx_exp2((-_cvt_f32_148.x) * 1.442695f);
                    float e1_99 = approx_exp2((-_cvt_f32_148.y) * 1.442695f);
                    float d0_99 = 1.0f + e0_99;
                    float d1_99 = 1.0f + e1_99;
                    float r0_99 = approx_rcp(d0_99);
                    float r1_99 = approx_rcp(d1_99);
                    float _fma_239;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(-d0_99), "f"(r0_99), "f"(1.0f));
                    float _fma_34_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34_1) : "f"(r0_99), "f"(_fma_239), "f"(r0_99));
                    float _fma_240;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(-d1_99), "f"(r1_99), "f"(1.0f));
                    float _fma_35_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35_1) : "f"(r1_99), "f"(_fma_240), "f"(r1_99));
                    float t0_99 = _fma_34_1;
                    float t1_99 = _fma_35_1;
                    float d4_99a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_99a) : "f"(d0_99), "f"(0.25f));
                    float r4_99a = approx_rcp(d4_99a);
                    float _fma_242;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(-d4_99a), "f"(r4_99a), "f"(1.0f));
                    float _fma_241;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(r4_99a), "f"(_fma_242), "f"(r4_99a));
                    float big_99a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_99a) : "f"(mul_f32_rn_noftz(_fma_241, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_243;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(big_99a), "f"(8.388608e+06f));
                    float whole_99a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_99a) : "f"(_add_243), "f"(8.388608e+06f));
                    float above_99a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_99a) : "f"(-d4_99a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_99a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_245;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_245) : "f"(big_99a), "f"(whole_99a));
                    float _abs_244;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_244) : "f"(_sub_245));
                    float _add_246;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(big_99a), "f"(0.5f));
                    float _sub_247;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_247) : "f"(big_99a), "f"(0.5f));
                    float rounded_99a = ((_abs_244 == 0.5f) ? ((above_99a > 0.0f) ? _add_246 : _sub_247) : whole_99a);
                    float u0_99 = ((d0_99 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_99a, 7.888609e-31f), 1.7763568e-15f) : t0_99);
                    float d4_99b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_99b) : "f"(d1_99), "f"(0.25f));
                    float r4_99b = approx_rcp(d4_99b);
                    float _fma_249;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(-d4_99b), "f"(r4_99b), "f"(1.0f));
                    float _fma_248;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(r4_99b), "f"(_fma_249), "f"(r4_99b));
                    float big_99b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_99b) : "f"(mul_f32_rn_noftz(_fma_248, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_250;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(big_99b), "f"(8.388608e+06f));
                    float whole_99b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_99b) : "f"(_add_250), "f"(8.388608e+06f));
                    float above_99b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_99b) : "f"(-d4_99b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_99b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_252;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_252) : "f"(big_99b), "f"(whole_99b));
                    float _abs_251;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_251) : "f"(_sub_252));
                    float _add_253;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(big_99b), "f"(0.5f));
                    float _sub_254;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_254) : "f"(big_99b), "f"(0.5f));
                    float rounded_99b = ((_abs_251 == 0.5f) ? ((above_99b > 0.0f) ? _add_253 : _sub_254) : whole_99b);
                    float u1_99 = ((d1_99 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_99b, 7.888609e-31f), 1.7763568e-15f) : t1_99);
                    float2 s_98 = make_float2(max_noftz(u0_99, 0.0f), max_noftz(u1_99, 0.0f));
                    float2 act_98;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_98) : "l"(*(const unsigned long long*)&_cvt_f32_148), "l"(*(const unsigned long long*)&s_98));
                    float2 _f32x2_mul_a_255 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_98, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_98), s_98), _cvt_f32_150);
                    float2 _mul_98;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_255), "l"(*(const unsigned long long*)&_cvt_f32_149));
                    float2 _mul_99;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_99) : "l"(*(const unsigned long long*)&act_98), "l"(*(const unsigned long long*)&_cvt_f32_150));
                    __nv_bfloat162 _bf16x2_98 = __float22bfloat162_rn(make_float2(_mul_98.x, _mul_98.y));
                    __nv_bfloat162 _bf16x2_99 = __float22bfloat162_rn(make_float2(_mul_99.x, _mul_99.y));
                    unsigned int dgate_new_1 = __as_u32(_bf16x2_98);
                    unsigned int dup_new_1 = __as_u32(_bf16x2_99);
                    dgate_b[0] = ((i_1 == 0) ? dgate_new_1 : dgate_b[0]);
                    dup_b[0] = ((i_1 == 0) ? dup_new_1 : dup_b[0]);
                    dgate_b[1] = ((i_1 == 1) ? dgate_new_1 : dgate_b[1]);
                    dup_b[1] = ((i_1 == 1) ? dup_new_1 : dup_b[1]);
                    dgate_b[2] = ((i_1 == 2) ? dgate_new_1 : dgate_b[2]);
                    dup_b[2] = ((i_1 == 2) ? dup_new_1 : dup_b[2]);
                    dgate_b[3] = ((i_1 == 3) ? dgate_new_1 : dgate_b[3]);
                    dup_b[3] = ((i_1 == 3) ? dup_new_1 : dup_b[3]);
                    dgate_b[4] = ((i_1 == 4) ? dgate_new_1 : dgate_b[4]);
                    dup_b[4] = ((i_1 == 4) ? dup_new_1 : dup_b[4]);
                    dgate_b[5] = ((i_1 == 5) ? dgate_new_1 : dgate_b[5]);
                    dup_b[5] = ((i_1 == 5) ? dup_new_1 : dup_b[5]);
                    dgate_b[6] = ((i_1 == 6) ? dgate_new_1 : dgate_b[6]);
                    dup_b[6] = ((i_1 == 6) ? dup_new_1 : dup_b[6]);
                    dgate_b[7] = ((i_1 == 7) ? dgate_new_1 : dgate_b[7]);
                    dup_b[7] = ((i_1 == 7) ? dup_new_1 : dup_b[7]);
                }
            }
        }
        int band_1 = 0;
        float2 _cvt_f32_151 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
        float2 _cvt_f32_152 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
        float2 _cvt_f32_153 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
        float e0_101 = approx_exp2((-_cvt_f32_151.x) * 1.442695f);
        float e1_101 = approx_exp2((-_cvt_f32_151.y) * 1.442695f);
        float d0_101 = 1.0f + e0_101;
        float d1_101 = 1.0f + e1_101;
        float r0_101 = approx_rcp(d0_101);
        float r1_101 = approx_rcp(d1_101);
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(-d0_101), "f"(r0_101), "f"(1.0f));
        float _fma_36_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36_1) : "f"(r0_101), "f"(_fma_256), "f"(r0_101));
        float _fma_257;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(-d1_101), "f"(r1_101), "f"(1.0f));
        float _fma_37_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37_1) : "f"(r1_101), "f"(_fma_257), "f"(r1_101));
        float t0_101 = _fma_36_1;
        float t1_101 = _fma_37_1;
        band_1 = band_1 | ((fminf(t0_101, t1_101) == 0.0f) ? 1 : 0);
        float2 s_100 = make_float2(max_noftz(t0_101, 0.0f), max_noftz(t1_101, 0.0f));
        float2 act_100;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_100) : "l"(*(const unsigned long long*)&_cvt_f32_151), "l"(*(const unsigned long long*)&s_100));
        float2 _f32x2_mul_a_258 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_100, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_100), s_100), _cvt_f32_153);
        float2 _mul_100;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_258), "l"(*(const unsigned long long*)&_cvt_f32_152));
        float2 _mul_101;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_101) : "l"(*(const unsigned long long*)&act_100), "l"(*(const unsigned long long*)&_cvt_f32_153));
        __nv_bfloat162 _bf16x2_100 = __float22bfloat162_rn(make_float2(_mul_100.x, _mul_100.y));
        __nv_bfloat162 _bf16x2_101 = __float22bfloat162_rn(make_float2(_mul_101.x, _mul_101.y));
        dgate_c[0] = __as_u32(_bf16x2_100);
        dup_c[0] = __as_u32(_bf16x2_101);
        float2 _cvt_f32_154 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[1]));
        float2 _cvt_f32_155 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[1]));
        float2 _cvt_f32_156 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[1]));
        float e0_103 = approx_exp2((-_cvt_f32_154.x) * 1.442695f);
        float e1_103 = approx_exp2((-_cvt_f32_154.y) * 1.442695f);
        float d0_103 = 1.0f + e0_103;
        float d1_103 = 1.0f + e1_103;
        float r0_103 = approx_rcp(d0_103);
        float r1_103 = approx_rcp(d1_103);
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d0_103), "f"(r0_103), "f"(1.0f));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r0_103), "f"(_fma_259), "f"(r0_103));
        float _fma_260;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(-d1_103), "f"(r1_103), "f"(1.0f));
        float _fma_39_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39_1) : "f"(r1_103), "f"(_fma_260), "f"(r1_103));
        float t0_103 = _fma_38;
        float t1_103 = _fma_39_1;
        band_1 = band_1 | ((fminf(t0_103, t1_103) == 0.0f) ? 2 : 0);
        float2 s_102 = make_float2(max_noftz(t0_103, 0.0f), max_noftz(t1_103, 0.0f));
        float2 act_102;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_102) : "l"(*(const unsigned long long*)&_cvt_f32_154), "l"(*(const unsigned long long*)&s_102));
        float2 _f32x2_mul_a_261 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_102, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_102), s_102), _cvt_f32_156);
        float2 _mul_102;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_261), "l"(*(const unsigned long long*)&_cvt_f32_155));
        float2 _mul_103;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_103) : "l"(*(const unsigned long long*)&act_102), "l"(*(const unsigned long long*)&_cvt_f32_156));
        __nv_bfloat162 _bf16x2_102 = __float22bfloat162_rn(make_float2(_mul_102.x, _mul_102.y));
        __nv_bfloat162 _bf16x2_103 = __float22bfloat162_rn(make_float2(_mul_103.x, _mul_103.y));
        dgate_c[1] = __as_u32(_bf16x2_102);
        dup_c[1] = __as_u32(_bf16x2_103);
        float2 _cvt_f32_157 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[2]));
        float2 _cvt_f32_158 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[2]));
        float2 _cvt_f32_159 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[2]));
        float e0_105 = approx_exp2((-_cvt_f32_157.x) * 1.442695f);
        float e1_105 = approx_exp2((-_cvt_f32_157.y) * 1.442695f);
        float d0_105 = 1.0f + e0_105;
        float d1_105 = 1.0f + e1_105;
        float r0_105 = approx_rcp(d0_105);
        float r1_105 = approx_rcp(d1_105);
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d0_105), "f"(r0_105), "f"(1.0f));
        float _fma_40_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40_1) : "f"(r0_105), "f"(_fma_262), "f"(r0_105));
        float _fma_263;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(-d1_105), "f"(r1_105), "f"(1.0f));
        float _fma_41_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41_1) : "f"(r1_105), "f"(_fma_263), "f"(r1_105));
        float t0_105 = _fma_40_1;
        float t1_105 = _fma_41_1;
        band_1 = band_1 | ((fminf(t0_105, t1_105) == 0.0f) ? 4 : 0);
        float2 s_104 = make_float2(max_noftz(t0_105, 0.0f), max_noftz(t1_105, 0.0f));
        float2 act_104;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_104) : "l"(*(const unsigned long long*)&_cvt_f32_157), "l"(*(const unsigned long long*)&s_104));
        float2 _f32x2_mul_a_264 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_104, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_104), s_104), _cvt_f32_159);
        float2 _mul_104;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_264), "l"(*(const unsigned long long*)&_cvt_f32_158));
        float2 _mul_105;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_105) : "l"(*(const unsigned long long*)&act_104), "l"(*(const unsigned long long*)&_cvt_f32_159));
        __nv_bfloat162 _bf16x2_104 = __float22bfloat162_rn(make_float2(_mul_104.x, _mul_104.y));
        __nv_bfloat162 _bf16x2_105 = __float22bfloat162_rn(make_float2(_mul_105.x, _mul_105.y));
        dgate_c[2] = __as_u32(_bf16x2_104);
        dup_c[2] = __as_u32(_bf16x2_105);
        float2 _cvt_f32_160 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[3]));
        float2 _cvt_f32_161 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[3]));
        float2 _cvt_f32_162 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[3]));
        float e0_107 = approx_exp2((-_cvt_f32_160.x) * 1.442695f);
        float e1_107 = approx_exp2((-_cvt_f32_160.y) * 1.442695f);
        float d0_107 = 1.0f + e0_107;
        float d1_107 = 1.0f + e1_107;
        float r0_107 = approx_rcp(d0_107);
        float r1_107 = approx_rcp(d1_107);
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(-d0_107), "f"(r0_107), "f"(1.0f));
        float _fma_42_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42_1) : "f"(r0_107), "f"(_fma_265), "f"(r0_107));
        float _fma_266;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(-d1_107), "f"(r1_107), "f"(1.0f));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(r1_107), "f"(_fma_266), "f"(r1_107));
        float t0_107 = _fma_42_1;
        float t1_107 = _fma_43;
        band_1 = band_1 | ((fminf(t0_107, t1_107) == 0.0f) ? 8 : 0);
        float2 s_106 = make_float2(max_noftz(t0_107, 0.0f), max_noftz(t1_107, 0.0f));
        float2 act_106;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_106) : "l"(*(const unsigned long long*)&_cvt_f32_160), "l"(*(const unsigned long long*)&s_106));
        float2 _f32x2_mul_a_267 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_106, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_106), s_106), _cvt_f32_162);
        float2 _mul_106;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&_cvt_f32_161));
        float2 _mul_107;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_107) : "l"(*(const unsigned long long*)&act_106), "l"(*(const unsigned long long*)&_cvt_f32_162));
        __nv_bfloat162 _bf16x2_106 = __float22bfloat162_rn(make_float2(_mul_106.x, _mul_106.y));
        __nv_bfloat162 _bf16x2_107 = __float22bfloat162_rn(make_float2(_mul_107.x, _mul_107.y));
        dgate_c[3] = __as_u32(_bf16x2_106);
        dup_c[3] = __as_u32(_bf16x2_107);
        float2 _cvt_f32_163 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[4]));
        float2 _cvt_f32_164 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[4]));
        float2 _cvt_f32_165 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[4]));
        float e0_109 = approx_exp2((-_cvt_f32_163.x) * 1.442695f);
        float e1_109 = approx_exp2((-_cvt_f32_163.y) * 1.442695f);
        float d0_109 = 1.0f + e0_109;
        float d1_109 = 1.0f + e1_109;
        float r0_109 = approx_rcp(d0_109);
        float r1_109 = approx_rcp(d1_109);
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d0_109), "f"(r0_109), "f"(1.0f));
        float _fma_44_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44_1) : "f"(r0_109), "f"(_fma_268), "f"(r0_109));
        float _fma_269;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(-d1_109), "f"(r1_109), "f"(1.0f));
        float _fma_45_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45_1) : "f"(r1_109), "f"(_fma_269), "f"(r1_109));
        float t0_109 = _fma_44_1;
        float t1_109 = _fma_45_1;
        band_1 = band_1 | ((fminf(t0_109, t1_109) == 0.0f) ? 16 : 0);
        float2 s_108 = make_float2(max_noftz(t0_109, 0.0f), max_noftz(t1_109, 0.0f));
        float2 act_108;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_108) : "l"(*(const unsigned long long*)&_cvt_f32_163), "l"(*(const unsigned long long*)&s_108));
        float2 _f32x2_mul_a_270 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_108, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_108), s_108), _cvt_f32_165);
        float2 _mul_108;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_270), "l"(*(const unsigned long long*)&_cvt_f32_164));
        float2 _mul_109;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_109) : "l"(*(const unsigned long long*)&act_108), "l"(*(const unsigned long long*)&_cvt_f32_165));
        __nv_bfloat162 _bf16x2_108 = __float22bfloat162_rn(make_float2(_mul_108.x, _mul_108.y));
        __nv_bfloat162 _bf16x2_109 = __float22bfloat162_rn(make_float2(_mul_109.x, _mul_109.y));
        dgate_c[4] = __as_u32(_bf16x2_108);
        dup_c[4] = __as_u32(_bf16x2_109);
        float2 _cvt_f32_166 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[5]));
        float2 _cvt_f32_167 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[5]));
        float2 _cvt_f32_168 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[5]));
        float e0_111 = approx_exp2((-_cvt_f32_166.x) * 1.442695f);
        float e1_111 = approx_exp2((-_cvt_f32_166.y) * 1.442695f);
        float d0_111 = 1.0f + e0_111;
        float d1_111 = 1.0f + e1_111;
        float r0_111 = approx_rcp(d0_111);
        float r1_111 = approx_rcp(d1_111);
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(-d0_111), "f"(r0_111), "f"(1.0f));
        float _fma_46_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46_1) : "f"(r0_111), "f"(_fma_271), "f"(r0_111));
        float _fma_272;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(-d1_111), "f"(r1_111), "f"(1.0f));
        float _fma_47_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47_1) : "f"(r1_111), "f"(_fma_272), "f"(r1_111));
        float t0_111 = _fma_46_1;
        float t1_111 = _fma_47_1;
        band_1 = band_1 | ((fminf(t0_111, t1_111) == 0.0f) ? 32 : 0);
        float2 s_110 = make_float2(max_noftz(t0_111, 0.0f), max_noftz(t1_111, 0.0f));
        float2 act_110;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_110) : "l"(*(const unsigned long long*)&_cvt_f32_166), "l"(*(const unsigned long long*)&s_110));
        float2 _f32x2_mul_a_273 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_110, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_110), s_110), _cvt_f32_168);
        float2 _mul_110;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_273), "l"(*(const unsigned long long*)&_cvt_f32_167));
        float2 _mul_111;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_111) : "l"(*(const unsigned long long*)&act_110), "l"(*(const unsigned long long*)&_cvt_f32_168));
        __nv_bfloat162 _bf16x2_110 = __float22bfloat162_rn(make_float2(_mul_110.x, _mul_110.y));
        __nv_bfloat162 _bf16x2_111 = __float22bfloat162_rn(make_float2(_mul_111.x, _mul_111.y));
        dgate_c[5] = __as_u32(_bf16x2_110);
        dup_c[5] = __as_u32(_bf16x2_111);
        float2 _cvt_f32_169 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[6]));
        float2 _cvt_f32_170 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[6]));
        float2 _cvt_f32_171 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[6]));
        float e0_113 = approx_exp2((-_cvt_f32_169.x) * 1.442695f);
        float e1_113 = approx_exp2((-_cvt_f32_169.y) * 1.442695f);
        float d0_113 = 1.0f + e0_113;
        float d1_113 = 1.0f + e1_113;
        float r0_113 = approx_rcp(d0_113);
        float r1_113 = approx_rcp(d1_113);
        float _fma_274;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(-d0_113), "f"(r0_113), "f"(1.0f));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r0_113), "f"(_fma_274), "f"(r0_113));
        float _fma_275;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(-d1_113), "f"(r1_113), "f"(1.0f));
        float _fma_49_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49_1) : "f"(r1_113), "f"(_fma_275), "f"(r1_113));
        float t0_113 = _fma_48;
        float t1_113 = _fma_49_1;
        band_1 = band_1 | ((fminf(t0_113, t1_113) == 0.0f) ? 64 : 0);
        float2 s_112 = make_float2(max_noftz(t0_113, 0.0f), max_noftz(t1_113, 0.0f));
        float2 act_112;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_112) : "l"(*(const unsigned long long*)&_cvt_f32_169), "l"(*(const unsigned long long*)&s_112));
        float2 _f32x2_mul_a_276 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_112, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_112), s_112), _cvt_f32_171);
        float2 _mul_112;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_112) : "l"(*(const unsigned long long*)&_f32x2_mul_a_276), "l"(*(const unsigned long long*)&_cvt_f32_170));
        float2 _mul_113;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_113) : "l"(*(const unsigned long long*)&act_112), "l"(*(const unsigned long long*)&_cvt_f32_171));
        __nv_bfloat162 _bf16x2_112 = __float22bfloat162_rn(make_float2(_mul_112.x, _mul_112.y));
        __nv_bfloat162 _bf16x2_113 = __float22bfloat162_rn(make_float2(_mul_113.x, _mul_113.y));
        dgate_c[6] = __as_u32(_bf16x2_112);
        dup_c[6] = __as_u32(_bf16x2_113);
        float2 _cvt_f32_172 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[7]));
        float2 _cvt_f32_173 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[7]));
        float2 _cvt_f32_174 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[7]));
        float e0_115 = approx_exp2((-_cvt_f32_172.x) * 1.442695f);
        float e1_115 = approx_exp2((-_cvt_f32_172.y) * 1.442695f);
        float d0_115 = 1.0f + e0_115;
        float d1_115 = 1.0f + e1_115;
        float r0_115 = approx_rcp(d0_115);
        float r1_115 = approx_rcp(d1_115);
        float _fma_277;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(-d0_115), "f"(r0_115), "f"(1.0f));
        float _fma_50_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50_1) : "f"(r0_115), "f"(_fma_277), "f"(r0_115));
        float _fma_278;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(-d1_115), "f"(r1_115), "f"(1.0f));
        float _fma_51_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51_1) : "f"(r1_115), "f"(_fma_278), "f"(r1_115));
        float t0_115 = _fma_50_1;
        float t1_115 = _fma_51_1;
        band_1 = band_1 | ((fminf(t0_115, t1_115) == 0.0f) ? 128 : 0);
        float2 s_114 = make_float2(max_noftz(t0_115, 0.0f), max_noftz(t1_115, 0.0f));
        float2 act_114;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_114) : "l"(*(const unsigned long long*)&_cvt_f32_172), "l"(*(const unsigned long long*)&s_114));
        float2 _f32x2_mul_a_279 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_114, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_114), s_114), _cvt_f32_174);
        float2 _mul_114;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_114) : "l"(*(const unsigned long long*)&_f32x2_mul_a_279), "l"(*(const unsigned long long*)&_cvt_f32_173));
        float2 _mul_115;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_115) : "l"(*(const unsigned long long*)&act_114), "l"(*(const unsigned long long*)&_cvt_f32_174));
        __nv_bfloat162 _bf16x2_114 = __float22bfloat162_rn(make_float2(_mul_114.x, _mul_114.y));
        __nv_bfloat162 _bf16x2_115 = __float22bfloat162_rn(make_float2(_mul_115.x, _mul_115.y));
        dgate_c[7] = __as_u32(_bf16x2_114);
        dup_c[7] = __as_u32(_bf16x2_115);
        if (band_1 != 0) {
            #pragma unroll 1
            for (int i_2 = 0; i_2 < 8; i_2++) {
                if ((band_1 >> i_2 & 1) != 0) {
                    unsigned int _vec_load_18[1];
                    {
                        _vec_load_18[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_1 + 2 * i_2) + 0);
                    }
                    unsigned int _vec_load_19[1];
                    {
                        _vec_load_19[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_1 + K + 2 * i_2) + 0);
                    }
                    unsigned int _vec_load_20[1];
                    {
                        _vec_load_20[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_c * K + col + 2 * i_2) + 0);
                    }
                    float2 _cvt_f32_175 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_18[0]));
                    float2 _cvt_f32_176 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_19[0]));
                    float2 _cvt_f32_177 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_20[0]));
                    float e0_117 = approx_exp2((-_cvt_f32_175.x) * 1.442695f);
                    float e1_117 = approx_exp2((-_cvt_f32_175.y) * 1.442695f);
                    float d0_117 = 1.0f + e0_117;
                    float d1_117 = 1.0f + e1_117;
                    float r0_117 = approx_rcp(d0_117);
                    float r1_117 = approx_rcp(d1_117);
                    float _fma_280;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(-d0_117), "f"(r0_117), "f"(1.0f));
                    float _fma_52_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52_1) : "f"(r0_117), "f"(_fma_280), "f"(r0_117));
                    float _fma_281;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(-d1_117), "f"(r1_117), "f"(1.0f));
                    float _fma_53;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r1_117), "f"(_fma_281), "f"(r1_117));
                    float t0_117 = _fma_52_1;
                    float t1_117 = _fma_53;
                    float d4_117a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_117a) : "f"(d0_117), "f"(0.25f));
                    float r4_117a = approx_rcp(d4_117a);
                    float _fma_283;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(-d4_117a), "f"(r4_117a), "f"(1.0f));
                    float _fma_282;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(r4_117a), "f"(_fma_283), "f"(r4_117a));
                    float big_117a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_117a) : "f"(mul_f32_rn_noftz(_fma_282, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_284;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(big_117a), "f"(8.388608e+06f));
                    float whole_117a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_117a) : "f"(_add_284), "f"(8.388608e+06f));
                    float above_117a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_117a) : "f"(-d4_117a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_117a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_286;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(big_117a), "f"(whole_117a));
                    float _abs_285;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_285) : "f"(_sub_286));
                    float _add_287;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(big_117a), "f"(0.5f));
                    float _sub_288;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_288) : "f"(big_117a), "f"(0.5f));
                    float rounded_117a = ((_abs_285 == 0.5f) ? ((above_117a > 0.0f) ? _add_287 : _sub_288) : whole_117a);
                    float u0_117 = ((d0_117 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_117a, 7.888609e-31f), 1.7763568e-15f) : t0_117);
                    float d4_117b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_117b) : "f"(d1_117), "f"(0.25f));
                    float r4_117b = approx_rcp(d4_117b);
                    float _fma_290;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(-d4_117b), "f"(r4_117b), "f"(1.0f));
                    float _fma_289;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(r4_117b), "f"(_fma_290), "f"(r4_117b));
                    float big_117b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_117b) : "f"(mul_f32_rn_noftz(_fma_289, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_291;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(big_117b), "f"(8.388608e+06f));
                    float whole_117b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_117b) : "f"(_add_291), "f"(8.388608e+06f));
                    float above_117b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_117b) : "f"(-d4_117b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_117b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_293;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(big_117b), "f"(whole_117b));
                    float _abs_292;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_292) : "f"(_sub_293));
                    float _add_294;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(big_117b), "f"(0.5f));
                    float _sub_295;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(big_117b), "f"(0.5f));
                    float rounded_117b = ((_abs_292 == 0.5f) ? ((above_117b > 0.0f) ? _add_294 : _sub_295) : whole_117b);
                    float u1_117 = ((d1_117 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_117b, 7.888609e-31f), 1.7763568e-15f) : t1_117);
                    float2 s_116 = make_float2(max_noftz(u0_117, 0.0f), max_noftz(u1_117, 0.0f));
                    float2 act_116;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_116) : "l"(*(const unsigned long long*)&_cvt_f32_175), "l"(*(const unsigned long long*)&s_116));
                    float2 _f32x2_mul_a_296 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_116, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_116), s_116), _cvt_f32_177);
                    float2 _mul_116;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_296), "l"(*(const unsigned long long*)&_cvt_f32_176));
                    float2 _mul_117;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_117) : "l"(*(const unsigned long long*)&act_116), "l"(*(const unsigned long long*)&_cvt_f32_177));
                    __nv_bfloat162 _bf16x2_116 = __float22bfloat162_rn(make_float2(_mul_116.x, _mul_116.y));
                    __nv_bfloat162 _bf16x2_117 = __float22bfloat162_rn(make_float2(_mul_117.x, _mul_117.y));
                    unsigned int dgate_new_2 = __as_u32(_bf16x2_116);
                    unsigned int dup_new_2 = __as_u32(_bf16x2_117);
                    dgate_c[0] = ((i_2 == 0) ? dgate_new_2 : dgate_c[0]);
                    dup_c[0] = ((i_2 == 0) ? dup_new_2 : dup_c[0]);
                    dgate_c[1] = ((i_2 == 1) ? dgate_new_2 : dgate_c[1]);
                    dup_c[1] = ((i_2 == 1) ? dup_new_2 : dup_c[1]);
                    dgate_c[2] = ((i_2 == 2) ? dgate_new_2 : dgate_c[2]);
                    dup_c[2] = ((i_2 == 2) ? dup_new_2 : dup_c[2]);
                    dgate_c[3] = ((i_2 == 3) ? dgate_new_2 : dgate_c[3]);
                    dup_c[3] = ((i_2 == 3) ? dup_new_2 : dup_c[3]);
                    dgate_c[4] = ((i_2 == 4) ? dgate_new_2 : dgate_c[4]);
                    dup_c[4] = ((i_2 == 4) ? dup_new_2 : dup_c[4]);
                    dgate_c[5] = ((i_2 == 5) ? dgate_new_2 : dgate_c[5]);
                    dup_c[5] = ((i_2 == 5) ? dup_new_2 : dup_c[5]);
                    dgate_c[6] = ((i_2 == 6) ? dgate_new_2 : dgate_c[6]);
                    dup_c[6] = ((i_2 == 6) ? dup_new_2 : dup_c[6]);
                    dgate_c[7] = ((i_2 == 7) ? dgate_new_2 : dgate_c[7]);
                    dup_c[7] = ((i_2 == 7) ? dup_new_2 : dup_c[7]);
                }
            }
        }
        int band_2 = 0;
        float2 _cvt_f32_178 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
        float2 _cvt_f32_179 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
        float2 _cvt_f32_180 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
        float e0_119 = approx_exp2((-_cvt_f32_178.x) * 1.442695f);
        float e1_119 = approx_exp2((-_cvt_f32_178.y) * 1.442695f);
        float d0_119 = 1.0f + e0_119;
        float d1_119 = 1.0f + e1_119;
        float r0_119 = approx_rcp(d0_119);
        float r1_119 = approx_rcp(d1_119);
        float _fma_297;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(-d0_119), "f"(r0_119), "f"(1.0f));
        float _fma_54_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54_1) : "f"(r0_119), "f"(_fma_297), "f"(r0_119));
        float _fma_298;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(-d1_119), "f"(r1_119), "f"(1.0f));
        float _fma_55_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55_1) : "f"(r1_119), "f"(_fma_298), "f"(r1_119));
        float t0_119 = _fma_54_1;
        float t1_119 = _fma_55_1;
        band_2 = band_2 | ((fminf(t0_119, t1_119) == 0.0f) ? 1 : 0);
        float2 s_118 = make_float2(max_noftz(t0_119, 0.0f), max_noftz(t1_119, 0.0f));
        float2 act_118;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_118) : "l"(*(const unsigned long long*)&_cvt_f32_178), "l"(*(const unsigned long long*)&s_118));
        float2 _f32x2_mul_a_299 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_118, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_118), s_118), _cvt_f32_180);
        float2 _mul_118;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_118) : "l"(*(const unsigned long long*)&_f32x2_mul_a_299), "l"(*(const unsigned long long*)&_cvt_f32_179));
        float2 _mul_119;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_119) : "l"(*(const unsigned long long*)&act_118), "l"(*(const unsigned long long*)&_cvt_f32_180));
        __nv_bfloat162 _bf16x2_118 = __float22bfloat162_rn(make_float2(_mul_118.x, _mul_118.y));
        __nv_bfloat162 _bf16x2_119 = __float22bfloat162_rn(make_float2(_mul_119.x, _mul_119.y));
        dgate_d[0] = __as_u32(_bf16x2_118);
        dup_d[0] = __as_u32(_bf16x2_119);
        float2 _cvt_f32_181 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[1]));
        float2 _cvt_f32_182 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[1]));
        float2 _cvt_f32_183 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[1]));
        float e0_121 = approx_exp2((-_cvt_f32_181.x) * 1.442695f);
        float e1_121 = approx_exp2((-_cvt_f32_181.y) * 1.442695f);
        float d0_121 = 1.0f + e0_121;
        float d1_121 = 1.0f + e1_121;
        float r0_121 = approx_rcp(d0_121);
        float r1_121 = approx_rcp(d1_121);
        float _fma_300;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(-d0_121), "f"(r0_121), "f"(1.0f));
        float _fma_56_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56_1) : "f"(r0_121), "f"(_fma_300), "f"(r0_121));
        float _fma_301;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(-d1_121), "f"(r1_121), "f"(1.0f));
        float _fma_57_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57_1) : "f"(r1_121), "f"(_fma_301), "f"(r1_121));
        float t0_121 = _fma_56_1;
        float t1_121 = _fma_57_1;
        band_2 = band_2 | ((fminf(t0_121, t1_121) == 0.0f) ? 2 : 0);
        float2 s_120 = make_float2(max_noftz(t0_121, 0.0f), max_noftz(t1_121, 0.0f));
        float2 act_120;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_120) : "l"(*(const unsigned long long*)&_cvt_f32_181), "l"(*(const unsigned long long*)&s_120));
        float2 _f32x2_mul_a_302 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_120, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_120), s_120), _cvt_f32_183);
        float2 _mul_120;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_302), "l"(*(const unsigned long long*)&_cvt_f32_182));
        float2 _mul_121;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_121) : "l"(*(const unsigned long long*)&act_120), "l"(*(const unsigned long long*)&_cvt_f32_183));
        __nv_bfloat162 _bf16x2_120 = __float22bfloat162_rn(make_float2(_mul_120.x, _mul_120.y));
        __nv_bfloat162 _bf16x2_121 = __float22bfloat162_rn(make_float2(_mul_121.x, _mul_121.y));
        dgate_d[1] = __as_u32(_bf16x2_120);
        dup_d[1] = __as_u32(_bf16x2_121);
        float2 _cvt_f32_184 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[2]));
        float2 _cvt_f32_185 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[2]));
        float2 _cvt_f32_186 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[2]));
        float e0_123 = approx_exp2((-_cvt_f32_184.x) * 1.442695f);
        float e1_123 = approx_exp2((-_cvt_f32_184.y) * 1.442695f);
        float d0_123 = 1.0f + e0_123;
        float d1_123 = 1.0f + e1_123;
        float r0_123 = approx_rcp(d0_123);
        float r1_123 = approx_rcp(d1_123);
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d0_123), "f"(r0_123), "f"(1.0f));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(r0_123), "f"(_fma_303), "f"(r0_123));
        float _fma_304;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(-d1_123), "f"(r1_123), "f"(1.0f));
        float _fma_59_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59_1) : "f"(r1_123), "f"(_fma_304), "f"(r1_123));
        float t0_123 = _fma_58;
        float t1_123 = _fma_59_1;
        band_2 = band_2 | ((fminf(t0_123, t1_123) == 0.0f) ? 4 : 0);
        float2 s_122 = make_float2(max_noftz(t0_123, 0.0f), max_noftz(t1_123, 0.0f));
        float2 act_122;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_122) : "l"(*(const unsigned long long*)&_cvt_f32_184), "l"(*(const unsigned long long*)&s_122));
        float2 _f32x2_mul_a_305 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_122, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_122), s_122), _cvt_f32_186);
        float2 _mul_122;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_122) : "l"(*(const unsigned long long*)&_f32x2_mul_a_305), "l"(*(const unsigned long long*)&_cvt_f32_185));
        float2 _mul_123;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_123) : "l"(*(const unsigned long long*)&act_122), "l"(*(const unsigned long long*)&_cvt_f32_186));
        __nv_bfloat162 _bf16x2_122 = __float22bfloat162_rn(make_float2(_mul_122.x, _mul_122.y));
        __nv_bfloat162 _bf16x2_123 = __float22bfloat162_rn(make_float2(_mul_123.x, _mul_123.y));
        dgate_d[2] = __as_u32(_bf16x2_122);
        dup_d[2] = __as_u32(_bf16x2_123);
        float2 _cvt_f32_187 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[3]));
        float2 _cvt_f32_188 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[3]));
        float2 _cvt_f32_189 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[3]));
        float e0_125 = approx_exp2((-_cvt_f32_187.x) * 1.442695f);
        float e1_125 = approx_exp2((-_cvt_f32_187.y) * 1.442695f);
        float d0_125 = 1.0f + e0_125;
        float d1_125 = 1.0f + e1_125;
        float r0_125 = approx_rcp(d0_125);
        float r1_125 = approx_rcp(d1_125);
        float _fma_306;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(-d0_125), "f"(r0_125), "f"(1.0f));
        float _fma_60_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60_1) : "f"(r0_125), "f"(_fma_306), "f"(r0_125));
        float _fma_307;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(-d1_125), "f"(r1_125), "f"(1.0f));
        float _fma_61_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61_1) : "f"(r1_125), "f"(_fma_307), "f"(r1_125));
        float t0_125 = _fma_60_1;
        float t1_125 = _fma_61_1;
        band_2 = band_2 | ((fminf(t0_125, t1_125) == 0.0f) ? 8 : 0);
        float2 s_124 = make_float2(max_noftz(t0_125, 0.0f), max_noftz(t1_125, 0.0f));
        float2 act_124;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_124) : "l"(*(const unsigned long long*)&_cvt_f32_187), "l"(*(const unsigned long long*)&s_124));
        float2 _f32x2_mul_a_308 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_124, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_124), s_124), _cvt_f32_189);
        float2 _mul_124;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_124) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&_cvt_f32_188));
        float2 _mul_125;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_125) : "l"(*(const unsigned long long*)&act_124), "l"(*(const unsigned long long*)&_cvt_f32_189));
        __nv_bfloat162 _bf16x2_124 = __float22bfloat162_rn(make_float2(_mul_124.x, _mul_124.y));
        __nv_bfloat162 _bf16x2_125 = __float22bfloat162_rn(make_float2(_mul_125.x, _mul_125.y));
        dgate_d[3] = __as_u32(_bf16x2_124);
        dup_d[3] = __as_u32(_bf16x2_125);
        float2 _cvt_f32_190 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[4]));
        float2 _cvt_f32_191 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[4]));
        float2 _cvt_f32_192 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[4]));
        float e0_127 = approx_exp2((-_cvt_f32_190.x) * 1.442695f);
        float e1_127 = approx_exp2((-_cvt_f32_190.y) * 1.442695f);
        float d0_127 = 1.0f + e0_127;
        float d1_127 = 1.0f + e1_127;
        float r0_127 = approx_rcp(d0_127);
        float r1_127 = approx_rcp(d1_127);
        float _fma_309;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(-d0_127), "f"(r0_127), "f"(1.0f));
        float _fma_62_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62_1) : "f"(r0_127), "f"(_fma_309), "f"(r0_127));
        float _fma_310;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(-d1_127), "f"(r1_127), "f"(1.0f));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r1_127), "f"(_fma_310), "f"(r1_127));
        float t0_127 = _fma_62_1;
        float t1_127 = _fma_63;
        band_2 = band_2 | ((fminf(t0_127, t1_127) == 0.0f) ? 16 : 0);
        float2 s_126 = make_float2(max_noftz(t0_127, 0.0f), max_noftz(t1_127, 0.0f));
        float2 act_126;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_126) : "l"(*(const unsigned long long*)&_cvt_f32_190), "l"(*(const unsigned long long*)&s_126));
        float2 _f32x2_mul_a_311 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_126, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_126), s_126), _cvt_f32_192);
        float2 _mul_126;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_311), "l"(*(const unsigned long long*)&_cvt_f32_191));
        float2 _mul_127;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_127) : "l"(*(const unsigned long long*)&act_126), "l"(*(const unsigned long long*)&_cvt_f32_192));
        __nv_bfloat162 _bf16x2_126 = __float22bfloat162_rn(make_float2(_mul_126.x, _mul_126.y));
        __nv_bfloat162 _bf16x2_127 = __float22bfloat162_rn(make_float2(_mul_127.x, _mul_127.y));
        dgate_d[4] = __as_u32(_bf16x2_126);
        dup_d[4] = __as_u32(_bf16x2_127);
        float2 _cvt_f32_193 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[5]));
        float2 _cvt_f32_194 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[5]));
        float2 _cvt_f32_195 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[5]));
        float e0_129 = approx_exp2((-_cvt_f32_193.x) * 1.442695f);
        float e1_129 = approx_exp2((-_cvt_f32_193.y) * 1.442695f);
        float d0_129 = 1.0f + e0_129;
        float d1_129 = 1.0f + e1_129;
        float r0_129 = approx_rcp(d0_129);
        float r1_129 = approx_rcp(d1_129);
        float _fma_312;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(-d0_129), "f"(r0_129), "f"(1.0f));
        float _fma_64_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64_1) : "f"(r0_129), "f"(_fma_312), "f"(r0_129));
        float _fma_313;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(-d1_129), "f"(r1_129), "f"(1.0f));
        float _fma_65_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65_1) : "f"(r1_129), "f"(_fma_313), "f"(r1_129));
        float t0_129 = _fma_64_1;
        float t1_129 = _fma_65_1;
        band_2 = band_2 | ((fminf(t0_129, t1_129) == 0.0f) ? 32 : 0);
        float2 s_128 = make_float2(max_noftz(t0_129, 0.0f), max_noftz(t1_129, 0.0f));
        float2 act_128;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_128) : "l"(*(const unsigned long long*)&_cvt_f32_193), "l"(*(const unsigned long long*)&s_128));
        float2 _f32x2_mul_a_314 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_128, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_128), s_128), _cvt_f32_195);
        float2 _mul_128;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_128) : "l"(*(const unsigned long long*)&_f32x2_mul_a_314), "l"(*(const unsigned long long*)&_cvt_f32_194));
        float2 _mul_129;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_129) : "l"(*(const unsigned long long*)&act_128), "l"(*(const unsigned long long*)&_cvt_f32_195));
        __nv_bfloat162 _bf16x2_128 = __float22bfloat162_rn(make_float2(_mul_128.x, _mul_128.y));
        __nv_bfloat162 _bf16x2_129 = __float22bfloat162_rn(make_float2(_mul_129.x, _mul_129.y));
        dgate_d[5] = __as_u32(_bf16x2_128);
        dup_d[5] = __as_u32(_bf16x2_129);
        float2 _cvt_f32_196 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[6]));
        float2 _cvt_f32_197 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[6]));
        float2 _cvt_f32_198 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[6]));
        float e0_131 = approx_exp2((-_cvt_f32_196.x) * 1.442695f);
        float e1_131 = approx_exp2((-_cvt_f32_196.y) * 1.442695f);
        float d0_131 = 1.0f + e0_131;
        float d1_131 = 1.0f + e1_131;
        float r0_131 = approx_rcp(d0_131);
        float r1_131 = approx_rcp(d1_131);
        float _fma_315;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(-d0_131), "f"(r0_131), "f"(1.0f));
        float _fma_66_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66_1) : "f"(r0_131), "f"(_fma_315), "f"(r0_131));
        float _fma_316;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(-d1_131), "f"(r1_131), "f"(1.0f));
        float _fma_67_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67_1) : "f"(r1_131), "f"(_fma_316), "f"(r1_131));
        float t0_131 = _fma_66_1;
        float t1_131 = _fma_67_1;
        band_2 = band_2 | ((fminf(t0_131, t1_131) == 0.0f) ? 64 : 0);
        float2 s_130 = make_float2(max_noftz(t0_131, 0.0f), max_noftz(t1_131, 0.0f));
        float2 act_130;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_130) : "l"(*(const unsigned long long*)&_cvt_f32_196), "l"(*(const unsigned long long*)&s_130));
        float2 _f32x2_mul_a_317 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_130, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_130), s_130), _cvt_f32_198);
        float2 _mul_130;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_130) : "l"(*(const unsigned long long*)&_f32x2_mul_a_317), "l"(*(const unsigned long long*)&_cvt_f32_197));
        float2 _mul_131;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_131) : "l"(*(const unsigned long long*)&act_130), "l"(*(const unsigned long long*)&_cvt_f32_198));
        __nv_bfloat162 _bf16x2_130 = __float22bfloat162_rn(make_float2(_mul_130.x, _mul_130.y));
        __nv_bfloat162 _bf16x2_131 = __float22bfloat162_rn(make_float2(_mul_131.x, _mul_131.y));
        dgate_d[6] = __as_u32(_bf16x2_130);
        dup_d[6] = __as_u32(_bf16x2_131);
        float2 _cvt_f32_199 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[7]));
        float2 _cvt_f32_200 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[7]));
        float2 _cvt_f32_201 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[7]));
        float e0_133 = approx_exp2((-_cvt_f32_199.x) * 1.442695f);
        float e1_133 = approx_exp2((-_cvt_f32_199.y) * 1.442695f);
        float d0_133 = 1.0f + e0_133;
        float d1_133 = 1.0f + e1_133;
        float r0_133 = approx_rcp(d0_133);
        float r1_133 = approx_rcp(d1_133);
        float _fma_318;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(-d0_133), "f"(r0_133), "f"(1.0f));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(r0_133), "f"(_fma_318), "f"(r0_133));
        float _fma_319;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(-d1_133), "f"(r1_133), "f"(1.0f));
        float _fma_69_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69_1) : "f"(r1_133), "f"(_fma_319), "f"(r1_133));
        float t0_133 = _fma_68;
        float t1_133 = _fma_69_1;
        band_2 = band_2 | ((fminf(t0_133, t1_133) == 0.0f) ? 128 : 0);
        float2 s_132 = make_float2(max_noftz(t0_133, 0.0f), max_noftz(t1_133, 0.0f));
        float2 act_132;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_132) : "l"(*(const unsigned long long*)&_cvt_f32_199), "l"(*(const unsigned long long*)&s_132));
        float2 _f32x2_mul_a_320 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_132, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_132), s_132), _cvt_f32_201);
        float2 _mul_132;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_132) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&_cvt_f32_200));
        float2 _mul_133;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_133) : "l"(*(const unsigned long long*)&act_132), "l"(*(const unsigned long long*)&_cvt_f32_201));
        __nv_bfloat162 _bf16x2_132 = __float22bfloat162_rn(make_float2(_mul_132.x, _mul_132.y));
        __nv_bfloat162 _bf16x2_133 = __float22bfloat162_rn(make_float2(_mul_133.x, _mul_133.y));
        dgate_d[7] = __as_u32(_bf16x2_132);
        dup_d[7] = __as_u32(_bf16x2_133);
        if (band_2 != 0) {
            #pragma unroll 1
            for (int i_3 = 0; i_3 < 8; i_3++) {
                if ((band_2 >> i_3 & 1) != 0) {
                    unsigned int _vec_load_21[1];
                    {
                        _vec_load_21[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_2 + 2 * i_3) + 0);
                    }
                    unsigned int _vec_load_22[1];
                    {
                        _vec_load_22[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_2 + K + 2 * i_3) + 0);
                    }
                    unsigned int _vec_load_23[1];
                    {
                        _vec_load_23[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_d * K + col + 2 * i_3) + 0);
                    }
                    float2 _cvt_f32_202 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_21[0]));
                    float2 _cvt_f32_203 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_22[0]));
                    float2 _cvt_f32_204 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_23[0]));
                    float e0_135 = approx_exp2((-_cvt_f32_202.x) * 1.442695f);
                    float e1_135 = approx_exp2((-_cvt_f32_202.y) * 1.442695f);
                    float d0_135 = 1.0f + e0_135;
                    float d1_135 = 1.0f + e1_135;
                    float r0_135 = approx_rcp(d0_135);
                    float r1_135 = approx_rcp(d1_135);
                    float _fma_321;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(-d0_135), "f"(r0_135), "f"(1.0f));
                    float _fma_70_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70_1) : "f"(r0_135), "f"(_fma_321), "f"(r0_135));
                    float _fma_322;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(-d1_135), "f"(r1_135), "f"(1.0f));
                    float _fma_71_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71_1) : "f"(r1_135), "f"(_fma_322), "f"(r1_135));
                    float t0_135 = _fma_70_1;
                    float t1_135 = _fma_71_1;
                    float d4_135a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_135a) : "f"(d0_135), "f"(0.25f));
                    float r4_135a = approx_rcp(d4_135a);
                    float _fma_324;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(-d4_135a), "f"(r4_135a), "f"(1.0f));
                    float _fma_323;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(r4_135a), "f"(_fma_324), "f"(r4_135a));
                    float big_135a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_135a) : "f"(mul_f32_rn_noftz(_fma_323, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_325;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(big_135a), "f"(8.388608e+06f));
                    float whole_135a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_135a) : "f"(_add_325), "f"(8.388608e+06f));
                    float above_135a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_135a) : "f"(-d4_135a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_135a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_327;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(big_135a), "f"(whole_135a));
                    float _abs_326;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_326) : "f"(_sub_327));
                    float _add_328;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(big_135a), "f"(0.5f));
                    float _sub_329;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(big_135a), "f"(0.5f));
                    float rounded_135a = ((_abs_326 == 0.5f) ? ((above_135a > 0.0f) ? _add_328 : _sub_329) : whole_135a);
                    float u0_135 = ((d0_135 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_135a, 7.888609e-31f), 1.7763568e-15f) : t0_135);
                    float d4_135b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_135b) : "f"(d1_135), "f"(0.25f));
                    float r4_135b = approx_rcp(d4_135b);
                    float _fma_331;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(-d4_135b), "f"(r4_135b), "f"(1.0f));
                    float _fma_330;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(r4_135b), "f"(_fma_331), "f"(r4_135b));
                    float big_135b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_135b) : "f"(mul_f32_rn_noftz(_fma_330, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_332;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(big_135b), "f"(8.388608e+06f));
                    float whole_135b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_135b) : "f"(_add_332), "f"(8.388608e+06f));
                    float above_135b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_135b) : "f"(-d4_135b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_135b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_334;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(big_135b), "f"(whole_135b));
                    float _abs_333;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_333) : "f"(_sub_334));
                    float _add_335;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(big_135b), "f"(0.5f));
                    float _sub_336;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(big_135b), "f"(0.5f));
                    float rounded_135b = ((_abs_333 == 0.5f) ? ((above_135b > 0.0f) ? _add_335 : _sub_336) : whole_135b);
                    float u1_135 = ((d1_135 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_135b, 7.888609e-31f), 1.7763568e-15f) : t1_135);
                    float2 s_134 = make_float2(max_noftz(u0_135, 0.0f), max_noftz(u1_135, 0.0f));
                    float2 act_134;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_134) : "l"(*(const unsigned long long*)&_cvt_f32_202), "l"(*(const unsigned long long*)&s_134));
                    float2 _f32x2_mul_a_337 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_134, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_134), s_134), _cvt_f32_204);
                    float2 _mul_134;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_134) : "l"(*(const unsigned long long*)&_f32x2_mul_a_337), "l"(*(const unsigned long long*)&_cvt_f32_203));
                    float2 _mul_135;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_135) : "l"(*(const unsigned long long*)&act_134), "l"(*(const unsigned long long*)&_cvt_f32_204));
                    __nv_bfloat162 _bf16x2_134 = __float22bfloat162_rn(make_float2(_mul_134.x, _mul_134.y));
                    __nv_bfloat162 _bf16x2_135 = __float22bfloat162_rn(make_float2(_mul_135.x, _mul_135.y));
                    unsigned int dgate_new_3 = __as_u32(_bf16x2_134);
                    unsigned int dup_new_3 = __as_u32(_bf16x2_135);
                    dgate_d[0] = ((i_3 == 0) ? dgate_new_3 : dgate_d[0]);
                    dup_d[0] = ((i_3 == 0) ? dup_new_3 : dup_d[0]);
                    dgate_d[1] = ((i_3 == 1) ? dgate_new_3 : dgate_d[1]);
                    dup_d[1] = ((i_3 == 1) ? dup_new_3 : dup_d[1]);
                    dgate_d[2] = ((i_3 == 2) ? dgate_new_3 : dgate_d[2]);
                    dup_d[2] = ((i_3 == 2) ? dup_new_3 : dup_d[2]);
                    dgate_d[3] = ((i_3 == 3) ? dgate_new_3 : dgate_d[3]);
                    dup_d[3] = ((i_3 == 3) ? dup_new_3 : dup_d[3]);
                    dgate_d[4] = ((i_3 == 4) ? dgate_new_3 : dgate_d[4]);
                    dup_d[4] = ((i_3 == 4) ? dup_new_3 : dup_d[4]);
                    dgate_d[5] = ((i_3 == 5) ? dgate_new_3 : dgate_d[5]);
                    dup_d[5] = ((i_3 == 5) ? dup_new_3 : dup_d[5]);
                    dgate_d[6] = ((i_3 == 6) ? dgate_new_3 : dgate_d[6]);
                    dup_d[6] = ((i_3 == 6) ? dup_new_3 : dup_d[6]);
                    dgate_d[7] = ((i_3 == 7) ? dgate_new_3 : dgate_d[7]);
                    dup_d[7] = ((i_3 == 7) ? dup_new_3 : dup_d[7]);
                }
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0 + 3072)
        : "memory");
    unsigned int strip_chunk1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src1 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src1 + 3072)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_339 = amax_a;
    __nv_bfloat162 _hxb_339 = __as_bf16x2(cols[0]);
    unsigned int _hxr_339;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_339) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_339)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_339)));
    __nv_bfloat162 _max_338 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_339);
    amax_a = _max_338;
    __nv_bfloat162 _hxa_341 = amax_b;
    __nv_bfloat162 _hxb_341 = __as_bf16x2(cols[16]);
    unsigned int _hxr_341;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_341) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_341)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_341)));
    __nv_bfloat162 _max_340 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_341);
    amax_b = _max_340;
    __nv_bfloat162 _hxa_343 = amax_a;
    __nv_bfloat162 _hxb_343 = __as_bf16x2(cols[1]);
    unsigned int _hxr_343;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_343) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_343)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_343)));
    __nv_bfloat162 _max_342 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_343);
    amax_a = _max_342;
    __nv_bfloat162 _hxa_345 = amax_b;
    __nv_bfloat162 _hxb_345 = __as_bf16x2(cols[17]);
    unsigned int _hxr_345;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_345) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_345)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_345)));
    __nv_bfloat162 _max_344 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_345);
    amax_b = _max_344;
    __nv_bfloat162 _hxa_347 = amax_a;
    __nv_bfloat162 _hxb_347 = __as_bf16x2(cols[2]);
    unsigned int _hxr_347;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_347) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_347)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_347)));
    __nv_bfloat162 _max_346 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_347);
    amax_a = _max_346;
    __nv_bfloat162 _hxa_349 = amax_b;
    __nv_bfloat162 _hxb_349 = __as_bf16x2(cols[18]);
    unsigned int _hxr_349;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_349) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_349)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_349)));
    __nv_bfloat162 _max_348 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_349);
    amax_b = _max_348;
    __nv_bfloat162 _hxa_351 = amax_a;
    __nv_bfloat162 _hxb_351 = __as_bf16x2(cols[3]);
    unsigned int _hxr_351;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_351) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_351)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_351)));
    __nv_bfloat162 _max_350 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_351);
    amax_a = _max_350;
    __nv_bfloat162 _hxa_353 = amax_b;
    __nv_bfloat162 _hxb_353 = __as_bf16x2(cols[19]);
    unsigned int _hxr_353;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_353) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_353)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_353)));
    __nv_bfloat162 _max_352 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_353);
    amax_b = _max_352;
    __nv_bfloat162 _hxa_355 = amax_a;
    __nv_bfloat162 _hxb_355 = __as_bf16x2(cols[4]);
    unsigned int _hxr_355;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_355) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_355)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_355)));
    __nv_bfloat162 _max_354 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_355);
    amax_a = _max_354;
    __nv_bfloat162 _hxa_357 = amax_b;
    __nv_bfloat162 _hxb_357 = __as_bf16x2(cols[20]);
    unsigned int _hxr_357;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_357) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_357)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_357)));
    __nv_bfloat162 _max_356 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_357);
    amax_b = _max_356;
    __nv_bfloat162 _hxa_359 = amax_a;
    __nv_bfloat162 _hxb_359 = __as_bf16x2(cols[5]);
    unsigned int _hxr_359;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_359) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_359)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_359)));
    __nv_bfloat162 _max_358 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_359);
    amax_a = _max_358;
    __nv_bfloat162 _hxa_361 = amax_b;
    __nv_bfloat162 _hxb_361 = __as_bf16x2(cols[21]);
    unsigned int _hxr_361;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_361) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_361)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_361)));
    __nv_bfloat162 _max_360 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_361);
    amax_b = _max_360;
    __nv_bfloat162 _hxa_363 = amax_a;
    __nv_bfloat162 _hxb_363 = __as_bf16x2(cols[6]);
    unsigned int _hxr_363;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_363) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_363)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_363)));
    __nv_bfloat162 _max_362 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_363);
    amax_a = _max_362;
    __nv_bfloat162 _hxa_365 = amax_b;
    __nv_bfloat162 _hxb_365 = __as_bf16x2(cols[22]);
    unsigned int _hxr_365;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_365) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_365)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_365)));
    __nv_bfloat162 _max_364 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_365);
    amax_b = _max_364;
    __nv_bfloat162 _hxa_367 = amax_a;
    __nv_bfloat162 _hxb_367 = __as_bf16x2(cols[7]);
    unsigned int _hxr_367;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_367) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_367)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_367)));
    __nv_bfloat162 _max_366 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_367);
    amax_a = _max_366;
    __nv_bfloat162 _hxa_369 = amax_b;
    __nv_bfloat162 _hxb_369 = __as_bf16x2(cols[23]);
    unsigned int _hxr_369;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_369) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_369)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_369)));
    __nv_bfloat162 _max_368 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_369);
    amax_b = _max_368;
    __nv_bfloat162 _hxa_371 = amax_a;
    __nv_bfloat162 _hxb_371 = __as_bf16x2(cols[8]);
    unsigned int _hxr_371;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_371) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_371)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_371)));
    __nv_bfloat162 _max_370 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_371);
    amax_a = _max_370;
    __nv_bfloat162 _hxa_373 = amax_b;
    __nv_bfloat162 _hxb_373 = __as_bf16x2(cols[24]);
    unsigned int _hxr_373;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_373) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_373)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_373)));
    __nv_bfloat162 _max_372 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_373);
    amax_b = _max_372;
    __nv_bfloat162 _hxa_375 = amax_a;
    __nv_bfloat162 _hxb_375 = __as_bf16x2(cols[9]);
    unsigned int _hxr_375;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_375) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_375)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_375)));
    __nv_bfloat162 _max_374 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_375);
    amax_a = _max_374;
    __nv_bfloat162 _hxa_377 = amax_b;
    __nv_bfloat162 _hxb_377 = __as_bf16x2(cols[25]);
    unsigned int _hxr_377;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_377) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_377)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_377)));
    __nv_bfloat162 _max_376 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_377);
    amax_b = _max_376;
    __nv_bfloat162 _hxa_379 = amax_a;
    __nv_bfloat162 _hxb_379 = __as_bf16x2(cols[10]);
    unsigned int _hxr_379;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_379) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_379)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_379)));
    __nv_bfloat162 _max_378 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_379);
    amax_a = _max_378;
    __nv_bfloat162 _hxa_381 = amax_b;
    __nv_bfloat162 _hxb_381 = __as_bf16x2(cols[26]);
    unsigned int _hxr_381;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_381) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_381)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_381)));
    __nv_bfloat162 _max_380 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_381);
    amax_b = _max_380;
    __nv_bfloat162 _hxa_383 = amax_a;
    __nv_bfloat162 _hxb_383 = __as_bf16x2(cols[11]);
    unsigned int _hxr_383;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_383) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_383)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_383)));
    __nv_bfloat162 _max_382 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_383);
    amax_a = _max_382;
    __nv_bfloat162 _hxa_385 = amax_b;
    __nv_bfloat162 _hxb_385 = __as_bf16x2(cols[27]);
    unsigned int _hxr_385;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_385) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_385)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_385)));
    __nv_bfloat162 _max_384 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_385);
    amax_b = _max_384;
    __nv_bfloat162 _hxa_387 = amax_a;
    __nv_bfloat162 _hxb_387 = __as_bf16x2(cols[12]);
    unsigned int _hxr_387;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_387) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_387)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_387)));
    __nv_bfloat162 _max_386 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_387);
    amax_a = _max_386;
    __nv_bfloat162 _hxa_389 = amax_b;
    __nv_bfloat162 _hxb_389 = __as_bf16x2(cols[28]);
    unsigned int _hxr_389;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_389) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_389)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_389)));
    __nv_bfloat162 _max_388 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_389);
    amax_b = _max_388;
    __nv_bfloat162 _hxa_391 = amax_a;
    __nv_bfloat162 _hxb_391 = __as_bf16x2(cols[13]);
    unsigned int _hxr_391;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_391) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_391)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_391)));
    __nv_bfloat162 _max_390 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_391);
    amax_a = _max_390;
    __nv_bfloat162 _hxa_393 = amax_b;
    __nv_bfloat162 _hxb_393 = __as_bf16x2(cols[29]);
    unsigned int _hxr_393;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_393) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_393)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_393)));
    __nv_bfloat162 _max_392 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_393);
    amax_b = _max_392;
    __nv_bfloat162 _hxa_395 = amax_a;
    __nv_bfloat162 _hxb_395 = __as_bf16x2(cols[14]);
    unsigned int _hxr_395;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_395) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_395)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_395)));
    __nv_bfloat162 _max_394 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_395);
    amax_a = _max_394;
    __nv_bfloat162 _hxa_397 = amax_b;
    __nv_bfloat162 _hxb_397 = __as_bf16x2(cols[30]);
    unsigned int _hxr_397;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_397) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_397)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_397)));
    __nv_bfloat162 _max_396 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_397);
    amax_b = _max_396;
    __nv_bfloat162 _hxa_399 = amax_a;
    __nv_bfloat162 _hxb_399 = __as_bf16x2(cols[15]);
    unsigned int _hxr_399;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_399) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_399)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_399)));
    __nv_bfloat162 _max_398 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_399);
    amax_a = _max_398;
    __nv_bfloat162 _hxa_401 = amax_b;
    __nv_bfloat162 _hxb_401 = __as_bf16x2(cols[31]);
    unsigned int _hxr_401;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_401) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_401)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_401)));
    __nv_bfloat162 _max_400 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_401);
    amax_b = _max_400;
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_403;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_402;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_402) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_403));
    words_a[0] = _mov_b32_pack_u16x2_402;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_405;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_404;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_404) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_405));
    words_b[0] = _mov_b32_pack_u16x2_404;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_407;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_406;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_406) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_407));
    words_a[1] = _mov_b32_pack_u16x2_406;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_409;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_408;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_408) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_409));
    words_b[1] = _mov_b32_pack_u16x2_408;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_411;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_410;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_410) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_411));
    words_a[2] = _mov_b32_pack_u16x2_410;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_413;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_412;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_412) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_413));
    words_b[2] = _mov_b32_pack_u16x2_412;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_415;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_414;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_414) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_415));
    words_a[3] = _mov_b32_pack_u16x2_414;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_417;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_416;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_416) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_417));
    words_b[3] = _mov_b32_pack_u16x2_416;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_419;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_418;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_418) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_419));
    words_a[4] = _mov_b32_pack_u16x2_418;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_421;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_420;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_420) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_421));
    words_b[4] = _mov_b32_pack_u16x2_420;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_423;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_422;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_422) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_423));
    words_a[5] = _mov_b32_pack_u16x2_422;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_425;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_424;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_424) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_425));
    words_b[5] = _mov_b32_pack_u16x2_424;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_427;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_426;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_426) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_427));
    words_a[6] = _mov_b32_pack_u16x2_426;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_429;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_428;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_428) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_429));
    words_b[6] = _mov_b32_pack_u16x2_428;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_431;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_431) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_431) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_430;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_430) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_431));
    words_a[7] = _mov_b32_pack_u16x2_430;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_433;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_433) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_433) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_432;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_432) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_433));
    words_b[7] = _mov_b32_pack_u16x2_432;
    {
        const unsigned* _raw_stv8_434 = reinterpret_cast<const unsigned*>(words_a + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a * M + out_row))), "r"(_raw_stv8_434[0]), "r"(_raw_stv8_434[1]), "r"(_raw_stv8_434[2]), "r"(_raw_stv8_434[3]), "r"(_raw_stv8_434[4]), "r"(_raw_stv8_434[5]), "r"(_raw_stv8_434[6]), "r"(_raw_stv8_434[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_435 = reinterpret_cast<const unsigned*>(words_b + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b * M + out_row))), "r"(_raw_stv8_435[0]), "r"(_raw_stv8_435[1]), "r"(_raw_stv8_435[2]), "r"(_raw_stv8_435[3]), "r"(_raw_stv8_435[4]), "r"(_raw_stv8_435[5]), "r"(_raw_stv8_435[6]), "r"(_raw_stv8_435[7]) : "memory");
    }
    int row_tiles = M >> 7;
    int rblk = by * 4 + (lane_1 & 3);
    int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
    int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_5);
    int lane_8 = tid & 31;
    int warp_9 = tid >> 5;
    int k8_10 = lane_8 & 7;
    int srow_11 = (k8_10 >> 1) * 32 + (lane_8 >> 3) * 2 + (k8_10 & 1);
    unsigned int sswizzle_12 = (unsigned int)(srow_11 >> 4 & 6 | srow_11 & 1);
    unsigned int cols_13[32];
    unsigned int strip_chunk0_14 = (unsigned int)(warp_9 * 2);
    unsigned int strip_src0_15 = tiles_addr + 16384 + (unsigned int)(srow_11 * 128) + ((strip_chunk0_14 ^ sswizzle_12) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[0]), "=r"(cols_13[1]), "=r"(cols_13[2]), "=r"(cols_13[3])
        : "r"(strip_src0_15)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[4]), "=r"(cols_13[5]), "=r"(cols_13[6]), "=r"(cols_13[7])
        : "r"(strip_src0_15 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[8]), "=r"(cols_13[9]), "=r"(cols_13[10]), "=r"(cols_13[11])
        : "r"(strip_src0_15 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[12]), "=r"(cols_13[13]), "=r"(cols_13[14]), "=r"(cols_13[15])
        : "r"(strip_src0_15 + 3072)
        : "memory");
    unsigned int strip_chunk1_16 = (unsigned int)(warp_9 * 2 + 1);
    unsigned int strip_src1_17 = tiles_addr + 16384 + (unsigned int)(srow_11 * 128) + ((strip_chunk1_16 ^ sswizzle_12) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[16]), "=r"(cols_13[17]), "=r"(cols_13[18]), "=r"(cols_13[19])
        : "r"(strip_src1_17)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[20]), "=r"(cols_13[21]), "=r"(cols_13[22]), "=r"(cols_13[23])
        : "r"(strip_src1_17 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[24]), "=r"(cols_13[25]), "=r"(cols_13[26]), "=r"(cols_13[27])
        : "r"(strip_src1_17 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[28]), "=r"(cols_13[29]), "=r"(cols_13[30]), "=r"(cols_13[31])
        : "r"(strip_src1_17 + 3072)
        : "memory");
    __nv_bfloat162 amax_a_18 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_19 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_437 = amax_a_18;
    __nv_bfloat162 _hxb_437 = __as_bf16x2(cols_13[0]);
    unsigned int _hxr_437;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_437) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_437)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_437)));
    __nv_bfloat162 _max_436 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_437);
    amax_a_18 = _max_436;
    __nv_bfloat162 _hxa_439 = amax_b_19;
    __nv_bfloat162 _hxb_439 = __as_bf16x2(cols_13[16]);
    unsigned int _hxr_439;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_439) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_439)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_439)));
    __nv_bfloat162 _max_438 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_439);
    amax_b_19 = _max_438;
    __nv_bfloat162 _hxa_441 = amax_a_18;
    __nv_bfloat162 _hxb_441 = __as_bf16x2(cols_13[1]);
    unsigned int _hxr_441;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_441) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_441)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_441)));
    __nv_bfloat162 _max_440 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_441);
    amax_a_18 = _max_440;
    __nv_bfloat162 _hxa_443 = amax_b_19;
    __nv_bfloat162 _hxb_443 = __as_bf16x2(cols_13[17]);
    unsigned int _hxr_443;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_443) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_443)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_443)));
    __nv_bfloat162 _max_442 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_443);
    amax_b_19 = _max_442;
    __nv_bfloat162 _hxa_445 = amax_a_18;
    __nv_bfloat162 _hxb_445 = __as_bf16x2(cols_13[2]);
    unsigned int _hxr_445;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_445) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_445)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_445)));
    __nv_bfloat162 _max_444 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_445);
    amax_a_18 = _max_444;
    __nv_bfloat162 _hxa_447 = amax_b_19;
    __nv_bfloat162 _hxb_447 = __as_bf16x2(cols_13[18]);
    unsigned int _hxr_447;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_447) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_447)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_447)));
    __nv_bfloat162 _max_446 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_447);
    amax_b_19 = _max_446;
    __nv_bfloat162 _hxa_449 = amax_a_18;
    __nv_bfloat162 _hxb_449 = __as_bf16x2(cols_13[3]);
    unsigned int _hxr_449;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_449) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_449)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_449)));
    __nv_bfloat162 _max_448 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_449);
    amax_a_18 = _max_448;
    __nv_bfloat162 _hxa_451 = amax_b_19;
    __nv_bfloat162 _hxb_451 = __as_bf16x2(cols_13[19]);
    unsigned int _hxr_451;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_451) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_451)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_451)));
    __nv_bfloat162 _max_450 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_451);
    amax_b_19 = _max_450;
    __nv_bfloat162 _hxa_453 = amax_a_18;
    __nv_bfloat162 _hxb_453 = __as_bf16x2(cols_13[4]);
    unsigned int _hxr_453;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_453) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_453)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_453)));
    __nv_bfloat162 _max_452 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_453);
    amax_a_18 = _max_452;
    __nv_bfloat162 _hxa_455 = amax_b_19;
    __nv_bfloat162 _hxb_455 = __as_bf16x2(cols_13[20]);
    unsigned int _hxr_455;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_455) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_455)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_455)));
    __nv_bfloat162 _max_454 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_455);
    amax_b_19 = _max_454;
    __nv_bfloat162 _hxa_457 = amax_a_18;
    __nv_bfloat162 _hxb_457 = __as_bf16x2(cols_13[5]);
    unsigned int _hxr_457;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_457) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_457)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_457)));
    __nv_bfloat162 _max_456 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_457);
    amax_a_18 = _max_456;
    __nv_bfloat162 _hxa_459 = amax_b_19;
    __nv_bfloat162 _hxb_459 = __as_bf16x2(cols_13[21]);
    unsigned int _hxr_459;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_459) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_459)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_459)));
    __nv_bfloat162 _max_458 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_459);
    amax_b_19 = _max_458;
    __nv_bfloat162 _hxa_461 = amax_a_18;
    __nv_bfloat162 _hxb_461 = __as_bf16x2(cols_13[6]);
    unsigned int _hxr_461;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_461) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_461)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_461)));
    __nv_bfloat162 _max_460 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_461);
    amax_a_18 = _max_460;
    __nv_bfloat162 _hxa_463 = amax_b_19;
    __nv_bfloat162 _hxb_463 = __as_bf16x2(cols_13[22]);
    unsigned int _hxr_463;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_463) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_463)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_463)));
    __nv_bfloat162 _max_462 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_463);
    amax_b_19 = _max_462;
    __nv_bfloat162 _hxa_465 = amax_a_18;
    __nv_bfloat162 _hxb_465 = __as_bf16x2(cols_13[7]);
    unsigned int _hxr_465;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_465) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_465)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_465)));
    __nv_bfloat162 _max_464 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_465);
    amax_a_18 = _max_464;
    __nv_bfloat162 _hxa_467 = amax_b_19;
    __nv_bfloat162 _hxb_467 = __as_bf16x2(cols_13[23]);
    unsigned int _hxr_467;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_467) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_467)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_467)));
    __nv_bfloat162 _max_466 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_467);
    amax_b_19 = _max_466;
    __nv_bfloat162 _hxa_469 = amax_a_18;
    __nv_bfloat162 _hxb_469 = __as_bf16x2(cols_13[8]);
    unsigned int _hxr_469;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_469) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_469)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_469)));
    __nv_bfloat162 _max_468 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_469);
    amax_a_18 = _max_468;
    __nv_bfloat162 _hxa_471 = amax_b_19;
    __nv_bfloat162 _hxb_471 = __as_bf16x2(cols_13[24]);
    unsigned int _hxr_471;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_471) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_471)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_471)));
    __nv_bfloat162 _max_470 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_471);
    amax_b_19 = _max_470;
    __nv_bfloat162 _hxa_473 = amax_a_18;
    __nv_bfloat162 _hxb_473 = __as_bf16x2(cols_13[9]);
    unsigned int _hxr_473;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_473) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_473)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_473)));
    __nv_bfloat162 _max_472 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_473);
    amax_a_18 = _max_472;
    __nv_bfloat162 _hxa_475 = amax_b_19;
    __nv_bfloat162 _hxb_475 = __as_bf16x2(cols_13[25]);
    unsigned int _hxr_475;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_475) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_475)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_475)));
    __nv_bfloat162 _max_474 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_475);
    amax_b_19 = _max_474;
    __nv_bfloat162 _hxa_477 = amax_a_18;
    __nv_bfloat162 _hxb_477 = __as_bf16x2(cols_13[10]);
    unsigned int _hxr_477;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_477) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_477)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_477)));
    __nv_bfloat162 _max_476 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_477);
    amax_a_18 = _max_476;
    __nv_bfloat162 _hxa_479 = amax_b_19;
    __nv_bfloat162 _hxb_479 = __as_bf16x2(cols_13[26]);
    unsigned int _hxr_479;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_479) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_479)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_479)));
    __nv_bfloat162 _max_478 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_479);
    amax_b_19 = _max_478;
    __nv_bfloat162 _hxa_481 = amax_a_18;
    __nv_bfloat162 _hxb_481 = __as_bf16x2(cols_13[11]);
    unsigned int _hxr_481;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_481) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_481)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_481)));
    __nv_bfloat162 _max_480 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_481);
    amax_a_18 = _max_480;
    __nv_bfloat162 _hxa_483 = amax_b_19;
    __nv_bfloat162 _hxb_483 = __as_bf16x2(cols_13[27]);
    unsigned int _hxr_483;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_483) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_483)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_483)));
    __nv_bfloat162 _max_482 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_483);
    amax_b_19 = _max_482;
    __nv_bfloat162 _hxa_485 = amax_a_18;
    __nv_bfloat162 _hxb_485 = __as_bf16x2(cols_13[12]);
    unsigned int _hxr_485;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_485) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_485)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_485)));
    __nv_bfloat162 _max_484 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_485);
    amax_a_18 = _max_484;
    __nv_bfloat162 _hxa_487 = amax_b_19;
    __nv_bfloat162 _hxb_487 = __as_bf16x2(cols_13[28]);
    unsigned int _hxr_487;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_487) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_487)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_487)));
    __nv_bfloat162 _max_486 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_487);
    amax_b_19 = _max_486;
    __nv_bfloat162 _hxa_489 = amax_a_18;
    __nv_bfloat162 _hxb_489 = __as_bf16x2(cols_13[13]);
    unsigned int _hxr_489;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_489) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_489)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_489)));
    __nv_bfloat162 _max_488 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_489);
    amax_a_18 = _max_488;
    __nv_bfloat162 _hxa_491 = amax_b_19;
    __nv_bfloat162 _hxb_491 = __as_bf16x2(cols_13[29]);
    unsigned int _hxr_491;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_491) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_491)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_491)));
    __nv_bfloat162 _max_490 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_491);
    amax_b_19 = _max_490;
    __nv_bfloat162 _hxa_493 = amax_a_18;
    __nv_bfloat162 _hxb_493 = __as_bf16x2(cols_13[14]);
    unsigned int _hxr_493;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_493) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_493)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_493)));
    __nv_bfloat162 _max_492 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_493);
    amax_a_18 = _max_492;
    __nv_bfloat162 _hxa_495 = amax_b_19;
    __nv_bfloat162 _hxb_495 = __as_bf16x2(cols_13[30]);
    unsigned int _hxr_495;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_495) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_495)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_495)));
    __nv_bfloat162 _max_494 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_495);
    amax_b_19 = _max_494;
    __nv_bfloat162 _hxa_497 = amax_a_18;
    __nv_bfloat162 _hxb_497 = __as_bf16x2(cols_13[15]);
    unsigned int _hxr_497;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_497) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_497)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_497)));
    __nv_bfloat162 _max_496 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_497);
    amax_a_18 = _max_496;
    __nv_bfloat162 _hxa_499 = amax_b_19;
    __nv_bfloat162 _hxb_499 = __as_bf16x2(cols_13[31]);
    unsigned int _hxr_499;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_499) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_499)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_499)));
    __nv_bfloat162 _max_498 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_499);
    amax_b_19 = _max_498;
    unsigned int mag_a_20 = __as_u32(amax_a_18) & 2147450879;
    unsigned int mag_b_21 = __as_u32(amax_b_19) & 2147450879;
    unsigned int lows_ab_22 = mag_a_20 & 65535 | mag_b_21 << 16;
    unsigned int highs_ab_23 = mag_a_20 >> 16 | mag_b_21 & 4294901760u;
    __nv_bfloat162 col_max_24 = __hmax2_nan(__as_bf16x2(lows_ab_22), __as_bf16x2(highs_ab_23));
    unsigned int col_max_bits_25 = __as_u32(col_max_24);
    unsigned int amax_bits_26 = (col_max_bits_25 & 65535) << 16;
    int scale_i32_27 = (int)(amax_bits_26 + 2031616 >> 23) - 8;
    if (scale_i32_27 < 0) {
        scale_i32_27 = 0;
    }
    if ((amax_bits_26 & 2139095040) == 2139095040) {
        scale_i32_27 = 255;
    }
    unsigned int scale_28 = (unsigned int)scale_i32_27;
    unsigned int inv_29 = 254 - scale_28 << 7;
    if (scale_28 == 255) {
        inv_29 = 32704;
    }
    unsigned int inv2_30 = inv_29 | inv_29 << 16;
    unsigned int amax_bits_31 = (col_max_bits_25 >> 16 & 65535) << 16;
    int scale_i32_32 = (int)(amax_bits_31 + 2031616 >> 23) - 8;
    if (scale_i32_32 < 0) {
        scale_i32_32 = 0;
    }
    if ((amax_bits_31 & 2139095040) == 2139095040) {
        scale_i32_32 = 255;
    }
    unsigned int scale_33 = (unsigned int)scale_i32_32;
    unsigned int inv_34 = 254 - scale_33 << 7;
    if (scale_33 == 255) {
        inv_34 = 32704;
    }
    unsigned int inv2_35 = inv_34 | inv_34 << 16;
    int gcol_a_36 = bx * 64 + K + warp_9 * 16 + (lane_8 >> 2);
    int gcol_b_37 = gcol_a_36 + 8;
    int out_row_38 = by * 128 + (lane_8 & 3) * 32;
    unsigned int words_a_39[8];
    unsigned int words_b_40[8];
    uint32_t _bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(cols_13[0]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(cols_13[1]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_501;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "r"(_bf16x2_mul_33));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_33);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_500;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_500) : "h"(_e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_501));
    words_a_39[0] = _mov_b32_pack_u16x2_500;
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(cols_13[16]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(cols_13[17]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_503;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "r"(_bf16x2_mul_35));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_35);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_502;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_502) : "h"(_e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_503));
    words_b_40[0] = _mov_b32_pack_u16x2_502;
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(cols_13[2]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(cols_13[3]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_505;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "r"(_bf16x2_mul_37));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_37);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_504;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_504) : "h"(_e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_505));
    words_a_39[1] = _mov_b32_pack_u16x2_504;
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(cols_13[18]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(cols_13[19]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_507;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "r"(_bf16x2_mul_39));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_39);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_506;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_506) : "h"(_e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_507));
    words_b_40[1] = _mov_b32_pack_u16x2_506;
    uint32_t _bf16x2_mul_40;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(cols_13[4]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(cols_13[5]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_509;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "r"(_bf16x2_mul_41));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_41);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_508;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_508) : "h"(_e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_509));
    words_a_39[2] = _mov_b32_pack_u16x2_508;
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(cols_13[20]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(cols_13[21]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_511;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "r"(_bf16x2_mul_43));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_43);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_510;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_510) : "h"(_e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_511));
    words_b_40[2] = _mov_b32_pack_u16x2_510;
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(cols_13[6]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(cols_13[7]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_513;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "r"(_bf16x2_mul_45));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_45);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_512;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_512) : "h"(_e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_513));
    words_a_39[3] = _mov_b32_pack_u16x2_512;
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(cols_13[22]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(cols_13[23]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_515;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "r"(_bf16x2_mul_47));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_47);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_514;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_514) : "h"(_e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_515));
    words_b_40[3] = _mov_b32_pack_u16x2_514;
    uint32_t _bf16x2_mul_48;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(cols_13[8]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(cols_13[9]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_517;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "r"(_bf16x2_mul_49));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_49);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_516;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_516) : "h"(_e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_517));
    words_a_39[4] = _mov_b32_pack_u16x2_516;
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(cols_13[24]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(cols_13[25]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_519;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "r"(_bf16x2_mul_51));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_51);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_518;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_518) : "h"(_e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_519));
    words_b_40[4] = _mov_b32_pack_u16x2_518;
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(cols_13[10]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(cols_13[11]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_521;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "r"(_bf16x2_mul_53));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_53);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_520;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_520) : "h"(_e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_521));
    words_a_39[5] = _mov_b32_pack_u16x2_520;
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(cols_13[26]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(cols_13[27]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_523;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "r"(_bf16x2_mul_55));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_55);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_522;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_522) : "h"(_e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_523));
    words_b_40[5] = _mov_b32_pack_u16x2_522;
    uint32_t _bf16x2_mul_56;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(cols_13[12]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(cols_13[13]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_525;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "r"(_bf16x2_mul_57));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_57);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_524;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_524) : "h"(_e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_525));
    words_a_39[6] = _mov_b32_pack_u16x2_524;
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(cols_13[28]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(cols_13[29]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_527;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_527) : "r"(_bf16x2_mul_59));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_59);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_527) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_526;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_526) : "h"(_e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_527));
    words_b_40[6] = _mov_b32_pack_u16x2_526;
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(cols_13[14]), "r"(inv2_30));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(cols_13[15]), "r"(inv2_30));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_529;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_529) : "r"(_bf16x2_mul_61));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_61);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_529) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_528;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_528) : "h"(_e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_529));
    words_a_39[7] = _mov_b32_pack_u16x2_528;
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(cols_13[30]), "r"(inv2_35));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(cols_13[31]), "r"(inv2_35));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_531;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_531) : "r"(_bf16x2_mul_63));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_63);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_531) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_530;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_530) : "h"(_e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_531));
    words_b_40[7] = _mov_b32_pack_u16x2_530;
    {
        const unsigned* _raw_stv8_532 = reinterpret_cast<const unsigned*>(words_a_39 + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a_36 * M + out_row_38))), "r"(_raw_stv8_532[0]), "r"(_raw_stv8_532[1]), "r"(_raw_stv8_532[2]), "r"(_raw_stv8_532[3]), "r"(_raw_stv8_532[4]), "r"(_raw_stv8_532[5]), "r"(_raw_stv8_532[6]), "r"(_raw_stv8_532[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_533 = reinterpret_cast<const unsigned*>(words_b_40 + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b_37 * M + out_row_38))), "r"(_raw_stv8_533[0]), "r"(_raw_stv8_533[1]), "r"(_raw_stv8_533[2]), "r"(_raw_stv8_533[3]), "r"(_raw_stv8_533[4]), "r"(_raw_stv8_533[5]), "r"(_raw_stv8_533[6]), "r"(_raw_stv8_533[7]) : "memory");
    }
    int row_tiles_41 = M >> 7;
    int rblk_42 = by * 4 + (lane_8 & 3);
    int col_scale_a_43 = ((gcol_a_36 >> 7) * row_tiles_41 + (rblk_42 >> 2)) * 512 + (gcol_a_36 & 31) * 16 + (gcol_a_36 >> 5 & 3) * 4 + (rblk_42 & 3);
    int col_scale_b_44 = ((gcol_b_37 >> 7) * row_tiles_41 + (rblk_42 >> 2)) * 512 + (gcol_b_37 & 31) * 16 + (gcol_b_37 >> 5 & 3) * 4 + (rblk_42 & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_43) + (0)) = (unsigned char)(scale_28);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_44) + (0)) = (unsigned char)(scale_33);
}

} // extern "C"
