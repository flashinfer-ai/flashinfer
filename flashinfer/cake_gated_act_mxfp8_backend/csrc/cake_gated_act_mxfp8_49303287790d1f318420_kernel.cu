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
    int gate_index_3 = grow_b * (2 * K) + col;
    unsigned int vec_load[8];
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_3 + (0))) : "memory");
        vec_load[0 + 0] = (uint32_t)_iv_3_0;
        vec_load[0 + 1] = (uint32_t)_iv_3_1;
        vec_load[0 + 2] = (uint32_t)_iv_3_2;
        vec_load[0 + 3] = (uint32_t)_iv_3_3;
        vec_load[0 + 4] = (uint32_t)_iv_3_4;
        vec_load[0 + 5] = (uint32_t)_iv_3_5;
        vec_load[0 + 6] = (uint32_t)_iv_3_6;
        vec_load[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int vec_load_1[8];
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_3 + K) + (0))) : "memory");
        vec_load_1[0 + 0] = (uint32_t)_iv_4_0;
        vec_load_1[0 + 1] = (uint32_t)_iv_4_1;
        vec_load_1[0 + 2] = (uint32_t)_iv_4_2;
        vec_load_1[0 + 3] = (uint32_t)_iv_4_3;
        vec_load_1[0 + 4] = (uint32_t)_iv_4_4;
        vec_load_1[0 + 5] = (uint32_t)_iv_4_5;
        vec_load_1[0 + 6] = (uint32_t)_iv_4_6;
        vec_load_1[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int vec_load_2[8];
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
        vec_load_2[0 + 0] = (uint32_t)_iv_5_0;
        vec_load_2[0 + 1] = (uint32_t)_iv_5_1;
        vec_load_2[0 + 2] = (uint32_t)_iv_5_2;
        vec_load_2[0 + 3] = (uint32_t)_iv_5_3;
        vec_load_2[0 + 4] = (uint32_t)_iv_5_4;
        vec_load_2[0 + 5] = (uint32_t)_iv_5_5;
        vec_load_2[0 + 6] = (uint32_t)_iv_5_6;
        vec_load_2[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_4 = grow_c * (2 * K) + col;
    unsigned int vec_load_0[8];
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
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_4 + (0))) : "memory");
        vec_load_0[0 + 0] = (uint32_t)_iv_6_0;
        vec_load_0[0 + 1] = (uint32_t)_iv_6_1;
        vec_load_0[0 + 2] = (uint32_t)_iv_6_2;
        vec_load_0[0 + 3] = (uint32_t)_iv_6_3;
        vec_load_0[0 + 4] = (uint32_t)_iv_6_4;
        vec_load_0[0 + 5] = (uint32_t)_iv_6_5;
        vec_load_0[0 + 6] = (uint32_t)_iv_6_6;
        vec_load_0[0 + 7] = (uint32_t)_iv_6_7;
    }
    unsigned int vec_load_1_0[8];
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
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_4 + K) + (0))) : "memory");
        vec_load_1_0[0 + 0] = (uint32_t)_iv_7_0;
        vec_load_1_0[0 + 1] = (uint32_t)_iv_7_1;
        vec_load_1_0[0 + 2] = (uint32_t)_iv_7_2;
        vec_load_1_0[0 + 3] = (uint32_t)_iv_7_3;
        vec_load_1_0[0 + 4] = (uint32_t)_iv_7_4;
        vec_load_1_0[0 + 5] = (uint32_t)_iv_7_5;
        vec_load_1_0[0 + 6] = (uint32_t)_iv_7_6;
        vec_load_1_0[0 + 7] = (uint32_t)_iv_7_7;
    }
    unsigned int vec_load_2_0[8];
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
        vec_load_2_0[0 + 0] = (uint32_t)_iv_8_0;
        vec_load_2_0[0 + 1] = (uint32_t)_iv_8_1;
        vec_load_2_0[0 + 2] = (uint32_t)_iv_8_2;
        vec_load_2_0[0 + 3] = (uint32_t)_iv_8_3;
        vec_load_2_0[0 + 4] = (uint32_t)_iv_8_4;
        vec_load_2_0[0 + 5] = (uint32_t)_iv_8_5;
        vec_load_2_0[0 + 6] = (uint32_t)_iv_8_6;
        vec_load_2_0[0 + 7] = (uint32_t)_iv_8_7;
    }
    int gate_index_5 = grow_d * (2 * K) + col;
    unsigned int vec_load_3[8];
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
            : "=r"(_iv_9_0), "=r"(_iv_9_1), "=r"(_iv_9_2), "=r"(_iv_9_3), "=r"(_iv_9_4), "=r"(_iv_9_5), "=r"(_iv_9_6), "=r"(_iv_9_7) : "l"((const void*)(gated_input + gate_index_5 + (0))) : "memory");
        vec_load_3[0 + 0] = (uint32_t)_iv_9_0;
        vec_load_3[0 + 1] = (uint32_t)_iv_9_1;
        vec_load_3[0 + 2] = (uint32_t)_iv_9_2;
        vec_load_3[0 + 3] = (uint32_t)_iv_9_3;
        vec_load_3[0 + 4] = (uint32_t)_iv_9_4;
        vec_load_3[0 + 5] = (uint32_t)_iv_9_5;
        vec_load_3[0 + 6] = (uint32_t)_iv_9_6;
        vec_load_3[0 + 7] = (uint32_t)_iv_9_7;
    }
    unsigned int vec_load_1_1[8];
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
            : "=r"(_iv_10_0), "=r"(_iv_10_1), "=r"(_iv_10_2), "=r"(_iv_10_3), "=r"(_iv_10_4), "=r"(_iv_10_5), "=r"(_iv_10_6), "=r"(_iv_10_7) : "l"((const void*)(gated_input + (gate_index_5 + K) + (0))) : "memory");
        vec_load_1_1[0 + 0] = (uint32_t)_iv_10_0;
        vec_load_1_1[0 + 1] = (uint32_t)_iv_10_1;
        vec_load_1_1[0 + 2] = (uint32_t)_iv_10_2;
        vec_load_1_1[0 + 3] = (uint32_t)_iv_10_3;
        vec_load_1_1[0 + 4] = (uint32_t)_iv_10_4;
        vec_load_1_1[0 + 5] = (uint32_t)_iv_10_5;
        vec_load_1_1[0 + 6] = (uint32_t)_iv_10_6;
        vec_load_1_1[0 + 7] = (uint32_t)_iv_10_7;
    }
    unsigned int vec_load_2_1[8];
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
        vec_load_2_1[0 + 0] = (uint32_t)_iv_11_0;
        vec_load_2_1[0 + 1] = (uint32_t)_iv_11_1;
        vec_load_2_1[0 + 2] = (uint32_t)_iv_11_2;
        vec_load_2_1[0 + 3] = (uint32_t)_iv_11_3;
        vec_load_2_1[0 + 4] = (uint32_t)_iv_11_4;
        vec_load_2_1[0 + 5] = (uint32_t)_iv_11_5;
        vec_load_2_1[0 + 6] = (uint32_t)_iv_11_6;
        vec_load_2_1[0 + 7] = (uint32_t)_iv_11_7;
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
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_13;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_12 = __all_sync(0xFFFFFFFF, _min_13 > -87.0f);
    if (_warp_vote_12 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0), "f"(_fma_15), "f"(r0));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r1), "f"(_fma_17), "f"(r1));
        float2 s = make_float2(_fma_14, _fma_16);
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), cvt_f32_2);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&cvt_f32_1));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&cvt_f32_2));
        __nv_bfloat162 bf16x2 = __float22bfloat162_rn(make_float2(mul.x, mul.y));
        __nv_bfloat162 bf16x2_1 = __float22bfloat162_rn(make_float2(mul_1.x, mul_1.y));
        dgate[0] = __as_u32(bf16x2);
        dup[0] = __as_u32(bf16x2_1);
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_0 = approx_exp2((-cvt_f32_0.x) * 1.442695f);
        float e1_0 = approx_exp2((-cvt_f32_0.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r0_0), "f"(_fma_20), "f"(r0_0));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r1_0), "f"(_fma_22), "f"(r1_0));
        float2 s_0 = make_float2(_fma_19, _fma_21);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), cvt_f32_2_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&cvt_f32_2_0));
        __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_0);
        dup[1] = __as_u32(bf16x2_1_0);
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_1 = approx_exp2((-cvt_f32_3.x) * 1.442695f);
        float e1_1 = approx_exp2((-cvt_f32_3.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_1), "f"(_fma_25), "f"(r0_1));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r1_1), "f"(_fma_27), "f"(r1_1));
        float2 s_1 = make_float2(_fma_24, _fma_26);
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), cvt_f32_2_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&cvt_f32_2_1));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_2);
        dup[2] = __as_u32(bf16x2_1_1);
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_2 = approx_exp2((-cvt_f32_4.x) * 1.442695f);
        float e1_2 = approx_exp2((-cvt_f32_4.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r0_2), "f"(_fma_30), "f"(r0_2));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r1_2), "f"(_fma_32), "f"(r1_2));
        float2 s_2 = make_float2(_fma_29, _fma_31);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), cvt_f32_2_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&cvt_f32_2_2));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_3);
        dup[3] = __as_u32(bf16x2_1_2);
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_3 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_3), "f"(_fma_35), "f"(r0_3));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r1_3), "f"(_fma_37), "f"(r1_3));
        float2 s_3 = make_float2(_fma_34, _fma_36);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&cvt_f32_2_3));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_4);
        dup[4] = __as_u32(bf16x2_1_3);
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_4 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r0_4), "f"(_fma_40), "f"(r0_4));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r1_4), "f"(_fma_42), "f"(r1_4));
        float2 s_4 = make_float2(_fma_39, _fma_41);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&cvt_f32_2_4));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_5);
        dup[5] = __as_u32(bf16x2_1_4);
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_5 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(r0_5), "f"(_fma_45), "f"(r0_5));
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r1_5), "f"(_fma_47), "f"(r1_5));
        float2 s_5 = make_float2(_fma_44, _fma_46);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_48 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&cvt_f32_2_5));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_6);
        dup[6] = __as_u32(bf16x2_1_5);
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_6 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
        float e1_6 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(r0_6), "f"(_fma_50), "f"(r0_6));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(r1_6), "f"(_fma_52), "f"(r1_6));
        float2 s_6 = make_float2(_fma_49, _fma_51);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), cvt_f32_2_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&cvt_f32_2_6));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_7);
        dup[7] = __as_u32(bf16x2_1_6);
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 cvt_f32_2_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_7 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(r0_7), "f"(_fma_55), "f"(r0_7));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r1_7), "f"(_fma_57), "f"(r1_7));
        float2 s_7 = make_float2(_fma_54, _fma_56);
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), cvt_f32_2_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&cvt_f32_2_7));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_b[0] = __as_u32(bf16x2_8);
        dup_b[0] = __as_u32(bf16x2_1_7);
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 cvt_f32_2_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_8 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(r0_8), "f"(_fma_60), "f"(r0_8));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(r1_8), "f"(_fma_62), "f"(r1_8));
        float2 s_8 = make_float2(_fma_59, _fma_61);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), cvt_f32_2_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&cvt_f32_2_8));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_b[1] = __as_u32(bf16x2_9);
        dup_b[1] = __as_u32(bf16x2_1_8);
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 cvt_f32_2_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_9 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r0_9), "f"(_fma_65), "f"(r0_9));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r1_9), "f"(_fma_67), "f"(r1_9));
        float2 s_9 = make_float2(_fma_64, _fma_66);
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), cvt_f32_2_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&cvt_f32_2_9));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_b[2] = __as_u32(bf16x2_10);
        dup_b[2] = __as_u32(bf16x2_1_9);
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 cvt_f32_2_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_10 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(r0_10), "f"(_fma_70), "f"(r0_10));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r1_10), "f"(_fma_72), "f"(r1_10));
        float2 s_10 = make_float2(_fma_69, _fma_71);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), cvt_f32_2_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&cvt_f32_2_10));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_b[3] = __as_u32(bf16x2_11);
        dup_b[3] = __as_u32(bf16x2_1_10);
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 cvt_f32_2_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_11 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r0_11), "f"(_fma_75), "f"(r0_11));
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(r1_11), "f"(_fma_77), "f"(r1_11));
        float2 s_11 = make_float2(_fma_74, _fma_76);
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), cvt_f32_2_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&cvt_f32_2_11));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_b[4] = __as_u32(bf16x2_12);
        dup_b[4] = __as_u32(bf16x2_1_11);
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 cvt_f32_2_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_12 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(r0_12), "f"(_fma_80), "f"(r0_12));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r1_12), "f"(_fma_82), "f"(r1_12));
        float2 s_12 = make_float2(_fma_79, _fma_81);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), cvt_f32_2_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&cvt_f32_2_12));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_b[5] = __as_u32(bf16x2_13);
        dup_b[5] = __as_u32(bf16x2_1_12);
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 cvt_f32_2_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_13 = approx_exp2((-cvt_f32_15.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_15.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(r0_13), "f"(_fma_85), "f"(r0_13));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(r1_13), "f"(_fma_87), "f"(r1_13));
        float2 s_13 = make_float2(_fma_84, _fma_86);
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), cvt_f32_2_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&cvt_f32_2_13));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_b[6] = __as_u32(bf16x2_14);
        dup_b[6] = __as_u32(bf16x2_1_13);
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 cvt_f32_2_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_14 = approx_exp2((-cvt_f32_16.x) * 1.442695f);
        float e1_14 = approx_exp2((-cvt_f32_16.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r0_14), "f"(_fma_90), "f"(r0_14));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r1_14), "f"(_fma_92), "f"(r1_14));
        float2 s_14 = make_float2(_fma_89, _fma_91);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), cvt_f32_2_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&cvt_f32_2_14));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_b[7] = __as_u32(bf16x2_15);
        dup_b[7] = __as_u32(bf16x2_1_14);
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 cvt_f32_2_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_15 = approx_exp2((-cvt_f32_17.x) * 1.442695f);
        float e1_15 = approx_exp2((-cvt_f32_17.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(r0_15), "f"(_fma_95), "f"(r0_15));
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(r1_15), "f"(_fma_97), "f"(r1_15));
        float2 s_15 = make_float2(_fma_94, _fma_96);
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&s_15));
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15), s_15), cvt_f32_2_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&cvt_f32_2_15));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate_c[0] = __as_u32(bf16x2_16);
        dup_c[0] = __as_u32(bf16x2_1_15);
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 cvt_f32_2_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_16 = approx_exp2((-cvt_f32_18.x) * 1.442695f);
        float e1_16 = approx_exp2((-cvt_f32_18.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(r0_16), "f"(_fma_100), "f"(r0_16));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(r1_16), "f"(_fma_102), "f"(r1_16));
        float2 s_16 = make_float2(_fma_99, _fma_101);
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), cvt_f32_2_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&cvt_f32_2_16));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate_c[1] = __as_u32(bf16x2_17);
        dup_c[1] = __as_u32(bf16x2_1_16);
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 cvt_f32_2_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_17 = approx_exp2((-cvt_f32_19.x) * 1.442695f);
        float e1_17 = approx_exp2((-cvt_f32_19.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(r0_17), "f"(_fma_105), "f"(r0_17));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(r1_17), "f"(_fma_107), "f"(r1_17));
        float2 s_17 = make_float2(_fma_104, _fma_106);
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_108 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), cvt_f32_2_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&cvt_f32_2_17));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate_c[2] = __as_u32(bf16x2_18);
        dup_c[2] = __as_u32(bf16x2_1_17);
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 cvt_f32_2_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_18 = approx_exp2((-cvt_f32_20.x) * 1.442695f);
        float e1_18 = approx_exp2((-cvt_f32_20.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(r0_18), "f"(_fma_110), "f"(r0_18));
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(r1_18), "f"(_fma_112), "f"(r1_18));
        float2 s_18 = make_float2(_fma_109, _fma_111);
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), cvt_f32_2_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&cvt_f32_2_18));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate_c[3] = __as_u32(bf16x2_19);
        dup_c[3] = __as_u32(bf16x2_1_18);
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 cvt_f32_2_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_19 = approx_exp2((-cvt_f32_21.x) * 1.442695f);
        float e1_19 = approx_exp2((-cvt_f32_21.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_114;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(r0_19), "f"(_fma_115), "f"(r0_19));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(r1_19), "f"(_fma_117), "f"(r1_19));
        float2 s_19 = make_float2(_fma_114, _fma_116);
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_118 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), cvt_f32_2_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&cvt_f32_2_19));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate_c[4] = __as_u32(bf16x2_20);
        dup_c[4] = __as_u32(bf16x2_1_19);
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 cvt_f32_2_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_20 = approx_exp2((-cvt_f32_22.x) * 1.442695f);
        float e1_20 = approx_exp2((-cvt_f32_22.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(r0_20), "f"(_fma_120), "f"(r0_20));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(r1_20), "f"(_fma_122), "f"(r1_20));
        float2 s_20 = make_float2(_fma_119, _fma_121);
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), cvt_f32_2_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&cvt_f32_2_20));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate_c[5] = __as_u32(bf16x2_21);
        dup_c[5] = __as_u32(bf16x2_1_20);
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 cvt_f32_2_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_21 = approx_exp2((-cvt_f32_23.x) * 1.442695f);
        float e1_21 = approx_exp2((-cvt_f32_23.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(r0_21), "f"(_fma_125), "f"(r0_21));
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(r1_21), "f"(_fma_127), "f"(r1_21));
        float2 s_21 = make_float2(_fma_124, _fma_126);
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_128 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), cvt_f32_2_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&cvt_f32_2_21));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate_c[6] = __as_u32(bf16x2_22);
        dup_c[6] = __as_u32(bf16x2_1_21);
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 cvt_f32_2_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_22 = approx_exp2((-cvt_f32_24.x) * 1.442695f);
        float e1_22 = approx_exp2((-cvt_f32_24.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float _fma_129;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(r0_22), "f"(_fma_130), "f"(r0_22));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(r1_22), "f"(_fma_132), "f"(r1_22));
        float2 s_22 = make_float2(_fma_129, _fma_131);
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), cvt_f32_2_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&cvt_f32_2_22));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_c[7] = __as_u32(bf16x2_23);
        dup_c[7] = __as_u32(bf16x2_1_22);
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[0]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 cvt_f32_2_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[0]));
        float e0_23 = approx_exp2((-cvt_f32_25.x) * 1.442695f);
        float e1_23 = approx_exp2((-cvt_f32_25.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(r0_23), "f"(_fma_135), "f"(r0_23));
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(r1_23), "f"(_fma_137), "f"(r1_23));
        float2 s_23 = make_float2(_fma_134, _fma_136);
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_138 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), cvt_f32_2_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&cvt_f32_2_23));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_d[0] = __as_u32(bf16x2_24);
        dup_d[0] = __as_u32(bf16x2_1_23);
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[1]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 cvt_f32_2_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[1]));
        float e0_24 = approx_exp2((-cvt_f32_26.x) * 1.442695f);
        float e1_24 = approx_exp2((-cvt_f32_26.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float _fma_139;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(r0_24), "f"(_fma_140), "f"(r0_24));
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(r1_24), "f"(_fma_142), "f"(r1_24));
        float2 s_24 = make_float2(_fma_139, _fma_141);
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), cvt_f32_2_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&cvt_f32_2_24));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_d[1] = __as_u32(bf16x2_25);
        dup_d[1] = __as_u32(bf16x2_1_24);
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[2]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 cvt_f32_2_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[2]));
        float e0_25 = approx_exp2((-cvt_f32_27.x) * 1.442695f);
        float e1_25 = approx_exp2((-cvt_f32_27.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_144;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(r0_25), "f"(_fma_145), "f"(r0_25));
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(r1_25), "f"(_fma_147), "f"(r1_25));
        float2 s_25 = make_float2(_fma_144, _fma_146);
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_148 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), cvt_f32_2_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&cvt_f32_2_25));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_d[2] = __as_u32(bf16x2_26);
        dup_d[2] = __as_u32(bf16x2_1_25);
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[3]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 cvt_f32_2_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[3]));
        float e0_26 = approx_exp2((-cvt_f32_28.x) * 1.442695f);
        float e1_26 = approx_exp2((-cvt_f32_28.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float _fma_149;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(r0_26), "f"(_fma_150), "f"(r0_26));
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(r1_26), "f"(_fma_152), "f"(r1_26));
        float2 s_26 = make_float2(_fma_149, _fma_151);
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), cvt_f32_2_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&cvt_f32_2_26));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_d[3] = __as_u32(bf16x2_27);
        dup_d[3] = __as_u32(bf16x2_1_26);
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[4]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 cvt_f32_2_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[4]));
        float e0_27 = approx_exp2((-cvt_f32_29.x) * 1.442695f);
        float e1_27 = approx_exp2((-cvt_f32_29.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_154;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(r0_27), "f"(_fma_155), "f"(r0_27));
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(r1_27), "f"(_fma_157), "f"(r1_27));
        float2 s_27 = make_float2(_fma_154, _fma_156);
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_158 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), cvt_f32_2_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&cvt_f32_2_27));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_d[4] = __as_u32(bf16x2_28);
        dup_d[4] = __as_u32(bf16x2_1_27);
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[5]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 cvt_f32_2_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[5]));
        float e0_28 = approx_exp2((-cvt_f32_30.x) * 1.442695f);
        float e1_28 = approx_exp2((-cvt_f32_30.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float _fma_159;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(r0_28), "f"(_fma_160), "f"(r0_28));
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(r1_28), "f"(_fma_162), "f"(r1_28));
        float2 s_28 = make_float2(_fma_159, _fma_161);
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_163 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), cvt_f32_2_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&cvt_f32_2_28));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_d[5] = __as_u32(bf16x2_29);
        dup_d[5] = __as_u32(bf16x2_1_28);
        float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[6]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 cvt_f32_2_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[6]));
        float e0_29 = approx_exp2((-cvt_f32_31.x) * 1.442695f);
        float e1_29 = approx_exp2((-cvt_f32_31.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_164;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(r0_29), "f"(_fma_165), "f"(r0_29));
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(r1_29), "f"(_fma_167), "f"(r1_29));
        float2 s_29 = make_float2(_fma_164, _fma_166);
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_168 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), cvt_f32_2_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&cvt_f32_2_29));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_d[6] = __as_u32(bf16x2_30);
        dup_d[6] = __as_u32(bf16x2_1_29);
        float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[7]));
        float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 cvt_f32_2_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[7]));
        float e0_30 = approx_exp2((-cvt_f32_32.x) * 1.442695f);
        float e1_30 = approx_exp2((-cvt_f32_32.y) * 1.442695f);
        float d0_30 = 1.0f + e0_30;
        float d1_30 = 1.0f + e1_30;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
        float _fma_169;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(r0_30), "f"(_fma_170), "f"(r0_30));
        float _fma_172;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(r1_30), "f"(_fma_172), "f"(r1_30));
        float2 s_30 = make_float2(_fma_169, _fma_171);
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), cvt_f32_2_30);
        float2 mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&cvt_f32_1_30));
        float2 mul_1_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_30) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&cvt_f32_2_30));
        __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_31.x, mul_31.y));
        __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_30.x, mul_1_30.y));
        dgate_d[7] = __as_u32(bf16x2_31);
        dup_d[7] = __as_u32(bf16x2_1_30);
    } else {
        int band = 0;
        float2 cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_31 = approx_exp2((-cvt_f32_33.x) * 1.442695f);
        float e1_31 = approx_exp2((-cvt_f32_33.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_31), "f"(_fma_174), "f"(r0_31));
        float _fma_175;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_31), "f"(_fma_175), "f"(r1_31));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_31 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31) : "l"(*(const unsigned long long*)&cvt_f32_33), "l"(*(const unsigned long long*)&s_31));
        float2 _f32x2_mul_a_176 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31), s_31), cvt_f32_2_31);
        float2 mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&cvt_f32_1_31));
        float2 mul_1_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31) : "l"(*(const unsigned long long*)&act_31), "l"(*(const unsigned long long*)&cvt_f32_2_31));
        __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_32.x, mul_32.y));
        __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_31.x, mul_1_31.y));
        dgate[0] = __as_u32(bf16x2_32);
        dup[0] = __as_u32(bf16x2_1_31);
        float2 cvt_f32_33_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_31_1 = approx_exp2((-cvt_f32_33_1.x) * 1.442695f);
        float e1_31_1 = approx_exp2((-cvt_f32_33_1.y) * 1.442695f);
        float d0_31_1 = 1.0f + e0_31_1;
        float d1_31_1 = 1.0f + e1_31_1;
        float r0_31_1 = approx_rcp(d0_31_1);
        float r1_31_1 = approx_rcp(d1_31_1);
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d0_31_1), "f"(r0_31_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_31_1), "f"(_fma_177), "f"(r0_31_1));
        float _fma_178;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(-d1_31_1), "f"(r1_31_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_31_1), "f"(_fma_178), "f"(r1_31_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_31_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31_1) : "l"(*(const unsigned long long*)&cvt_f32_33_1), "l"(*(const unsigned long long*)&s_31_1));
        float2 _f32x2_mul_a_179 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31_1), s_31_1), cvt_f32_2_31_1);
        float2 mul_32_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&cvt_f32_1_31_1));
        float2 mul_1_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31_1) : "l"(*(const unsigned long long*)&act_31_1), "l"(*(const unsigned long long*)&cvt_f32_2_31_1));
        __nv_bfloat162 bf16x2_32_1 = __float22bfloat162_rn(make_float2(mul_32_1.x, mul_32_1.y));
        __nv_bfloat162 bf16x2_1_31_1 = __float22bfloat162_rn(make_float2(mul_1_31_1.x, mul_1_31_1.y));
        dgate[1] = __as_u32(bf16x2_32_1);
        dup[1] = __as_u32(bf16x2_1_31_1);
        float2 cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_32 = approx_exp2((-cvt_f32_34.x) * 1.442695f);
        float e1_32 = approx_exp2((-cvt_f32_34.y) * 1.442695f);
        float d0_32 = 1.0f + e0_32;
        float d1_32 = 1.0f + e1_32;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d0_32), "f"(r0_32), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_32), "f"(_fma_180), "f"(r0_32));
        float _fma_181;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(-d1_32), "f"(r1_32), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_32), "f"(_fma_181), "f"(r1_32));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_32 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&cvt_f32_34), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_182 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), cvt_f32_2_32);
        float2 mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&cvt_f32_1_32));
        float2 mul_1_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_32) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&cvt_f32_2_32));
        __nv_bfloat162 bf16x2_33 = __float22bfloat162_rn(make_float2(mul_33.x, mul_33.y));
        __nv_bfloat162 bf16x2_1_32 = __float22bfloat162_rn(make_float2(mul_1_32.x, mul_1_32.y));
        dgate[2] = __as_u32(bf16x2_33);
        dup[2] = __as_u32(bf16x2_1_32);
        float2 cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_33 = approx_exp2((-cvt_f32_35.x) * 1.442695f);
        float e1_33 = approx_exp2((-cvt_f32_35.y) * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_33), "f"(_fma_183), "f"(r0_33));
        float _fma_184;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_33), "f"(_fma_184), "f"(r1_33));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_33 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_33) : "l"(*(const unsigned long long*)&cvt_f32_35), "l"(*(const unsigned long long*)&s_33));
        float2 _f32x2_mul_a_185 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_33, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_33), s_33), cvt_f32_2_33);
        float2 mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&cvt_f32_1_33));
        float2 mul_1_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_33) : "l"(*(const unsigned long long*)&act_33), "l"(*(const unsigned long long*)&cvt_f32_2_33));
        __nv_bfloat162 bf16x2_34 = __float22bfloat162_rn(make_float2(mul_34.x, mul_34.y));
        __nv_bfloat162 bf16x2_1_33 = __float22bfloat162_rn(make_float2(mul_1_33.x, mul_1_33.y));
        dgate[3] = __as_u32(bf16x2_34);
        dup[3] = __as_u32(bf16x2_1_33);
        float2 cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_34 = approx_exp2((-cvt_f32_36.x) * 1.442695f);
        float e1_34 = approx_exp2((-cvt_f32_36.y) * 1.442695f);
        float d0_34 = 1.0f + e0_34;
        float d1_34 = 1.0f + e1_34;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(-d0_34), "f"(r0_34), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_34), "f"(_fma_186), "f"(r0_34));
        float _fma_187;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(-d1_34), "f"(r1_34), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_34), "f"(_fma_187), "f"(r1_34));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 s_34 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_34) : "l"(*(const unsigned long long*)&cvt_f32_36), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_188 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_34, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), cvt_f32_2_34);
        float2 mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&cvt_f32_1_34));
        float2 mul_1_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_34) : "l"(*(const unsigned long long*)&act_34), "l"(*(const unsigned long long*)&cvt_f32_2_34));
        __nv_bfloat162 bf16x2_35 = __float22bfloat162_rn(make_float2(mul_35.x, mul_35.y));
        __nv_bfloat162 bf16x2_1_34 = __float22bfloat162_rn(make_float2(mul_1_34.x, mul_1_34.y));
        dgate[4] = __as_u32(bf16x2_35);
        dup[4] = __as_u32(bf16x2_1_34);
        float2 cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_35 = approx_exp2((-cvt_f32_37.x) * 1.442695f);
        float e1_35 = approx_exp2((-cvt_f32_37.y) * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_35), "f"(_fma_189), "f"(r0_35));
        float _fma_190;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_35), "f"(_fma_190), "f"(r1_35));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 s_35 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_35) : "l"(*(const unsigned long long*)&cvt_f32_37), "l"(*(const unsigned long long*)&s_35));
        float2 _f32x2_mul_a_191 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_35, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_35), s_35), cvt_f32_2_35);
        float2 mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&cvt_f32_1_35));
        float2 mul_1_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_35) : "l"(*(const unsigned long long*)&act_35), "l"(*(const unsigned long long*)&cvt_f32_2_35));
        __nv_bfloat162 bf16x2_36 = __float22bfloat162_rn(make_float2(mul_36.x, mul_36.y));
        __nv_bfloat162 bf16x2_1_35 = __float22bfloat162_rn(make_float2(mul_1_35.x, mul_1_35.y));
        dgate[5] = __as_u32(bf16x2_36);
        dup[5] = __as_u32(bf16x2_1_35);
        float2 cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_36 = approx_exp2((-cvt_f32_38.x) * 1.442695f);
        float e1_36 = approx_exp2((-cvt_f32_38.y) * 1.442695f);
        float d0_36 = 1.0f + e0_36;
        float d1_36 = 1.0f + e1_36;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(-d0_36), "f"(r0_36), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_36), "f"(_fma_192), "f"(r0_36));
        float _fma_193;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(-d1_36), "f"(r1_36), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_36), "f"(_fma_193), "f"(r1_36));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 s_36 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_36) : "l"(*(const unsigned long long*)&cvt_f32_38), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_194 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_36, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), cvt_f32_2_36);
        float2 mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&cvt_f32_1_36));
        float2 mul_1_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_36) : "l"(*(const unsigned long long*)&act_36), "l"(*(const unsigned long long*)&cvt_f32_2_36));
        __nv_bfloat162 bf16x2_37 = __float22bfloat162_rn(make_float2(mul_37.x, mul_37.y));
        __nv_bfloat162 bf16x2_1_36 = __float22bfloat162_rn(make_float2(mul_1_36.x, mul_1_36.y));
        dgate[6] = __as_u32(bf16x2_37);
        dup[6] = __as_u32(bf16x2_1_36);
        float2 cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_37 = approx_exp2((-cvt_f32_39.x) * 1.442695f);
        float e1_37 = approx_exp2((-cvt_f32_39.y) * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_195;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_37), "f"(_fma_195), "f"(r0_37));
        float _fma_196;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_37), "f"(_fma_196), "f"(r1_37));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 s_37 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_37) : "l"(*(const unsigned long long*)&cvt_f32_39), "l"(*(const unsigned long long*)&s_37));
        float2 _f32x2_mul_a_197 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_37, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_37), s_37), cvt_f32_2_37);
        float2 mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&cvt_f32_1_37));
        float2 mul_1_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_37) : "l"(*(const unsigned long long*)&act_37), "l"(*(const unsigned long long*)&cvt_f32_2_37));
        __nv_bfloat162 bf16x2_38 = __float22bfloat162_rn(make_float2(mul_38.x, mul_38.y));
        __nv_bfloat162 bf16x2_1_37 = __float22bfloat162_rn(make_float2(mul_1_37.x, mul_1_37.y));
        dgate[7] = __as_u32(bf16x2_38);
        dup[7] = __as_u32(bf16x2_1_37);
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
                    float2 cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_12[0]));
                    float2 cvt_f32_1_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_13[0]));
                    float2 cvt_f32_2_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_14[0]));
                    float e0_38 = approx_exp2((-cvt_f32_40.x) * 1.442695f);
                    float e1_38 = approx_exp2((-cvt_f32_40.y) * 1.442695f);
                    float d0_38 = 1.0f + e0_38;
                    float d1_38 = 1.0f + e1_38;
                    float r0_38 = approx_rcp(d0_38);
                    float r1_38 = approx_rcp(d1_38);
                    float _fma_198;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(-d0_38), "f"(r0_38), "f"(1.0f));
                    float fma_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_38), "f"(_fma_198), "f"(r0_38));
                    float _fma_199;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(-d1_38), "f"(r1_38), "f"(1.0f));
                    float fma_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_38), "f"(_fma_199), "f"(r1_38));
                    float t0_7 = fma_8;
                    float t1_7 = fma_1_7;
                    float d4_1a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_38), "f"(0.25f));
                    float r4_1a = approx_rcp(d4_1a);
                    float _fma_201;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
                    float _fma_200;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(r4_1a), "f"(_fma_201), "f"(r4_1a));
                    float big_1a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_200, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_202;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(big_1a), "f"(8.388608e+06f));
                    float whole_1a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_202), "f"(8.388608e+06f));
                    float above_1a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_204;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_204) : "f"(big_1a), "f"(whole_1a));
                    float _abs_203;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(_sub_204));
                    float _add_205;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(big_1a), "f"(0.5f));
                    float _sub_206;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_206) : "f"(big_1a), "f"(0.5f));
                    float rounded_1a = ((_abs_203 == 0.5f) ? ((above_1a > 0.0f) ? _add_205 : _sub_206) : whole_1a);
                    float u0 = ((d0_38 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0_7);
                    float d4_1a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_38), "f"(0.25f));
                    float r4_1a_0 = approx_rcp(d4_1a_0);
                    float _fma_208;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
                    float _fma_207;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(r4_1a_0), "f"(_fma_208), "f"(r4_1a_0));
                    float big_1a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_207, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_209;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(big_1a_0), "f"(8.388608e+06f));
                    float whole_1a_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_209), "f"(8.388608e+06f));
                    float above_1a_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_211;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_211) : "f"(big_1a_0), "f"(whole_1a_0));
                    float _abs_210;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(_sub_211));
                    float _add_212;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(big_1a_0), "f"(0.5f));
                    float _sub_213;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(big_1a_0), "f"(0.5f));
                    float rounded_1a_0 = ((_abs_210 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_212 : _sub_213) : whole_1a_0);
                    float u1 = ((d1_38 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1_7);
                    float2 s_38 = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
                    float2 act_38;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_38) : "l"(*(const unsigned long long*)&cvt_f32_40), "l"(*(const unsigned long long*)&s_38));
                    float2 _f32x2_mul_a_214 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_38, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), cvt_f32_2_38);
                    float2 mul_39;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&cvt_f32_1_38));
                    float2 mul_1_38;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_38) : "l"(*(const unsigned long long*)&act_38), "l"(*(const unsigned long long*)&cvt_f32_2_38));
                    __nv_bfloat162 bf16x2_39 = __float22bfloat162_rn(make_float2(mul_39.x, mul_39.y));
                    __nv_bfloat162 bf16x2_1_38 = __float22bfloat162_rn(make_float2(mul_1_38.x, mul_1_38.y));
                    unsigned int dgate_new = __as_u32(bf16x2_39);
                    unsigned int dup_new = __as_u32(bf16x2_1_38);
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
        float2 cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 cvt_f32_2_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_39 = approx_exp2((-cvt_f32_41.x) * 1.442695f);
        float e1_39 = approx_exp2((-cvt_f32_41.y) * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_39), "f"(_fma_215), "f"(r0_39));
        float _fma_216;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_39), "f"(_fma_216), "f"(r1_39));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        band_0 = band_0 | ((fminf(t0_8, t1_8) == 0.0f) ? 1 : 0);
        float2 s_39 = make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f));
        float2 act_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_39) : "l"(*(const unsigned long long*)&cvt_f32_41), "l"(*(const unsigned long long*)&s_39));
        float2 _f32x2_mul_a_217 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_39, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_39), s_39), cvt_f32_2_39);
        float2 mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_217), "l"(*(const unsigned long long*)&cvt_f32_1_39));
        float2 mul_1_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_39) : "l"(*(const unsigned long long*)&act_39), "l"(*(const unsigned long long*)&cvt_f32_2_39));
        __nv_bfloat162 bf16x2_40 = __float22bfloat162_rn(make_float2(mul_40.x, mul_40.y));
        __nv_bfloat162 bf16x2_1_39 = __float22bfloat162_rn(make_float2(mul_1_39.x, mul_1_39.y));
        dgate_b[0] = __as_u32(bf16x2_40);
        dup_b[0] = __as_u32(bf16x2_1_39);
        float2 cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 cvt_f32_2_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_40 = approx_exp2((-cvt_f32_42.x) * 1.442695f);
        float e1_40 = approx_exp2((-cvt_f32_42.y) * 1.442695f);
        float d0_40 = 1.0f + e0_40;
        float d1_40 = 1.0f + e1_40;
        float r0_40 = approx_rcp(d0_40);
        float r1_40 = approx_rcp(d1_40);
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d0_40), "f"(r0_40), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_40), "f"(_fma_218), "f"(r0_40));
        float _fma_219;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(-d1_40), "f"(r1_40), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_40), "f"(_fma_219), "f"(r1_40));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        band_0 = band_0 | ((fminf(t0_9, t1_9) == 0.0f) ? 2 : 0);
        float2 s_40 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_40) : "l"(*(const unsigned long long*)&cvt_f32_42), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_220 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_40, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), cvt_f32_2_40);
        float2 mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&cvt_f32_1_40));
        float2 mul_1_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_40) : "l"(*(const unsigned long long*)&act_40), "l"(*(const unsigned long long*)&cvt_f32_2_40));
        __nv_bfloat162 bf16x2_41 = __float22bfloat162_rn(make_float2(mul_41.x, mul_41.y));
        __nv_bfloat162 bf16x2_1_40 = __float22bfloat162_rn(make_float2(mul_1_40.x, mul_1_40.y));
        dgate_b[1] = __as_u32(bf16x2_41);
        dup_b[1] = __as_u32(bf16x2_1_40);
        float2 cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 cvt_f32_2_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_41 = approx_exp2((-cvt_f32_43.x) * 1.442695f);
        float e1_41 = approx_exp2((-cvt_f32_43.y) * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_41), "f"(_fma_221), "f"(r0_41));
        float _fma_222;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_41), "f"(_fma_222), "f"(r1_41));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        band_0 = band_0 | ((fminf(t0_10, t1_10) == 0.0f) ? 4 : 0);
        float2 s_41 = make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f));
        float2 act_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_41) : "l"(*(const unsigned long long*)&cvt_f32_43), "l"(*(const unsigned long long*)&s_41));
        float2 _f32x2_mul_a_223 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_41, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_41), s_41), cvt_f32_2_41);
        float2 mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&cvt_f32_1_41));
        float2 mul_1_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_41) : "l"(*(const unsigned long long*)&act_41), "l"(*(const unsigned long long*)&cvt_f32_2_41));
        __nv_bfloat162 bf16x2_42 = __float22bfloat162_rn(make_float2(mul_42.x, mul_42.y));
        __nv_bfloat162 bf16x2_1_41 = __float22bfloat162_rn(make_float2(mul_1_41.x, mul_1_41.y));
        dgate_b[2] = __as_u32(bf16x2_42);
        dup_b[2] = __as_u32(bf16x2_1_41);
        float2 cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 cvt_f32_2_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_42 = approx_exp2((-cvt_f32_44.x) * 1.442695f);
        float e1_42 = approx_exp2((-cvt_f32_44.y) * 1.442695f);
        float d0_42 = 1.0f + e0_42;
        float d1_42 = 1.0f + e1_42;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(-d0_42), "f"(r0_42), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_42), "f"(_fma_224), "f"(r0_42));
        float _fma_225;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(-d1_42), "f"(r1_42), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_42), "f"(_fma_225), "f"(r1_42));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        band_0 = band_0 | ((fminf(t0_11, t1_11) == 0.0f) ? 8 : 0);
        float2 s_42 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_42) : "l"(*(const unsigned long long*)&cvt_f32_44), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_226 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_42, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), cvt_f32_2_42);
        float2 mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&cvt_f32_1_42));
        float2 mul_1_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_42) : "l"(*(const unsigned long long*)&act_42), "l"(*(const unsigned long long*)&cvt_f32_2_42));
        __nv_bfloat162 bf16x2_43 = __float22bfloat162_rn(make_float2(mul_43.x, mul_43.y));
        __nv_bfloat162 bf16x2_1_42 = __float22bfloat162_rn(make_float2(mul_1_42.x, mul_1_42.y));
        dgate_b[3] = __as_u32(bf16x2_43);
        dup_b[3] = __as_u32(bf16x2_1_42);
        float2 cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 cvt_f32_2_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_43 = approx_exp2((-cvt_f32_45.x) * 1.442695f);
        float e1_43 = approx_exp2((-cvt_f32_45.y) * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_43), "f"(_fma_227), "f"(r0_43));
        float _fma_228;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_43), "f"(_fma_228), "f"(r1_43));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        band_0 = band_0 | ((fminf(t0_12, t1_12) == 0.0f) ? 16 : 0);
        float2 s_43 = make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f));
        float2 act_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_43) : "l"(*(const unsigned long long*)&cvt_f32_45), "l"(*(const unsigned long long*)&s_43));
        float2 _f32x2_mul_a_229 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_43, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_43), s_43), cvt_f32_2_43);
        float2 mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&cvt_f32_1_43));
        float2 mul_1_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_43) : "l"(*(const unsigned long long*)&act_43), "l"(*(const unsigned long long*)&cvt_f32_2_43));
        __nv_bfloat162 bf16x2_44 = __float22bfloat162_rn(make_float2(mul_44.x, mul_44.y));
        __nv_bfloat162 bf16x2_1_43 = __float22bfloat162_rn(make_float2(mul_1_43.x, mul_1_43.y));
        dgate_b[4] = __as_u32(bf16x2_44);
        dup_b[4] = __as_u32(bf16x2_1_43);
        float2 cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 cvt_f32_2_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_44 = approx_exp2((-cvt_f32_46.x) * 1.442695f);
        float e1_44 = approx_exp2((-cvt_f32_46.y) * 1.442695f);
        float d0_44 = 1.0f + e0_44;
        float d1_44 = 1.0f + e1_44;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d0_44), "f"(r0_44), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_44), "f"(_fma_230), "f"(r0_44));
        float _fma_231;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(-d1_44), "f"(r1_44), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_44), "f"(_fma_231), "f"(r1_44));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        band_0 = band_0 | ((fminf(t0_13, t1_13) == 0.0f) ? 32 : 0);
        float2 s_44 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_44) : "l"(*(const unsigned long long*)&cvt_f32_46), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_232 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_44, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), cvt_f32_2_44);
        float2 mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&cvt_f32_1_44));
        float2 mul_1_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_44) : "l"(*(const unsigned long long*)&act_44), "l"(*(const unsigned long long*)&cvt_f32_2_44));
        __nv_bfloat162 bf16x2_45 = __float22bfloat162_rn(make_float2(mul_45.x, mul_45.y));
        __nv_bfloat162 bf16x2_1_44 = __float22bfloat162_rn(make_float2(mul_1_44.x, mul_1_44.y));
        dgate_b[5] = __as_u32(bf16x2_45);
        dup_b[5] = __as_u32(bf16x2_1_44);
        float2 cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 cvt_f32_2_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_45 = approx_exp2((-cvt_f32_47.x) * 1.442695f);
        float e1_45 = approx_exp2((-cvt_f32_47.y) * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_45), "f"(_fma_233), "f"(r0_45));
        float _fma_234;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_45), "f"(_fma_234), "f"(r1_45));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        band_0 = band_0 | ((fminf(t0_14, t1_14) == 0.0f) ? 64 : 0);
        float2 s_45 = make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f));
        float2 act_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_45) : "l"(*(const unsigned long long*)&cvt_f32_47), "l"(*(const unsigned long long*)&s_45));
        float2 _f32x2_mul_a_235 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_45, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_45), s_45), cvt_f32_2_45);
        float2 mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&cvt_f32_1_45));
        float2 mul_1_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_45) : "l"(*(const unsigned long long*)&act_45), "l"(*(const unsigned long long*)&cvt_f32_2_45));
        __nv_bfloat162 bf16x2_46 = __float22bfloat162_rn(make_float2(mul_46.x, mul_46.y));
        __nv_bfloat162 bf16x2_1_45 = __float22bfloat162_rn(make_float2(mul_1_45.x, mul_1_45.y));
        dgate_b[6] = __as_u32(bf16x2_46);
        dup_b[6] = __as_u32(bf16x2_1_45);
        float2 cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 cvt_f32_2_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_46 = approx_exp2((-cvt_f32_48.x) * 1.442695f);
        float e1_46 = approx_exp2((-cvt_f32_48.y) * 1.442695f);
        float d0_46 = 1.0f + e0_46;
        float d1_46 = 1.0f + e1_46;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float _fma_236;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d0_46), "f"(r0_46), "f"(1.0f));
        float fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_16) : "f"(r0_46), "f"(_fma_236), "f"(r0_46));
        float _fma_237;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(-d1_46), "f"(r1_46), "f"(1.0f));
        float fma_1_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_15) : "f"(r1_46), "f"(_fma_237), "f"(r1_46));
        float t0_15 = fma_16;
        float t1_15 = fma_1_15;
        band_0 = band_0 | ((fminf(t0_15, t1_15) == 0.0f) ? 128 : 0);
        float2 s_46 = make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f));
        float2 act_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_46) : "l"(*(const unsigned long long*)&cvt_f32_48), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_238 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_46, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), cvt_f32_2_46);
        float2 mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&cvt_f32_1_46));
        float2 mul_1_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_46) : "l"(*(const unsigned long long*)&act_46), "l"(*(const unsigned long long*)&cvt_f32_2_46));
        __nv_bfloat162 bf16x2_47 = __float22bfloat162_rn(make_float2(mul_47.x, mul_47.y));
        __nv_bfloat162 bf16x2_1_46 = __float22bfloat162_rn(make_float2(mul_1_46.x, mul_1_46.y));
        dgate_b[7] = __as_u32(bf16x2_47);
        dup_b[7] = __as_u32(bf16x2_1_46);
        if (band_0 != 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 8; i_1++) {
                if ((band_0 >> i_1 & 1) != 0) {
                    unsigned int vec_load_4[1];
                    {
                        vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_3 + 2 * i_1) + 0);
                    }
                    unsigned int vec_load_1_2[1];
                    {
                        vec_load_1_2[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_3 + K + 2 * i_1) + 0);
                    }
                    unsigned int vec_load_2_2[1];
                    {
                        vec_load_2_2[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_b * K + col + 2 * i_1) + 0);
                    }
                    float2 cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[0]));
                    float2 cvt_f32_1_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[0]));
                    float2 cvt_f32_2_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[0]));
                    float e0_47 = approx_exp2((-cvt_f32_49.x) * 1.442695f);
                    float e1_47 = approx_exp2((-cvt_f32_49.y) * 1.442695f);
                    float d0_47 = 1.0f + e0_47;
                    float d1_47 = 1.0f + e1_47;
                    float r0_47 = approx_rcp(d0_47);
                    float r1_47 = approx_rcp(d1_47);
                    float _fma_239;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
                    float fma_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_17) : "f"(r0_47), "f"(_fma_239), "f"(r0_47));
                    float _fma_240;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
                    float fma_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_16) : "f"(r1_47), "f"(_fma_240), "f"(r1_47));
                    float t0_16 = fma_17;
                    float t1_16 = fma_1_16;
                    float d4_1a_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_47), "f"(0.25f));
                    float r4_1a_1 = approx_rcp(d4_1a_1);
                    float _fma_242;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
                    float _fma_241;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(r4_1a_1), "f"(_fma_242), "f"(r4_1a_1));
                    float big_1a_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_241, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_243;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(big_1a_1), "f"(8.388608e+06f));
                    float whole_1a_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_243), "f"(8.388608e+06f));
                    float above_1a_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_245;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_245) : "f"(big_1a_1), "f"(whole_1a_1));
                    float _abs_244;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_244) : "f"(_sub_245));
                    float _add_246;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(big_1a_1), "f"(0.5f));
                    float _sub_247;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_247) : "f"(big_1a_1), "f"(0.5f));
                    float rounded_1a_1 = ((_abs_244 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_246 : _sub_247) : whole_1a_1);
                    float u0_1 = ((d0_47 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                    float d4_1a_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_47), "f"(0.25f));
                    float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
                    float _fma_249;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
                    float _fma_248;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(r4_1a_1_1), "f"(_fma_249), "f"(r4_1a_1_1));
                    float big_1a_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_248, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_250;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(big_1a_1_1), "f"(8.388608e+06f));
                    float whole_1a_1_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_250), "f"(8.388608e+06f));
                    float above_1a_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_252;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_252) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
                    float _abs_251;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_251) : "f"(_sub_252));
                    float _add_253;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(big_1a_1_1), "f"(0.5f));
                    float _sub_254;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_254) : "f"(big_1a_1_1), "f"(0.5f));
                    float rounded_1a_1_1 = ((_abs_251 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_253 : _sub_254) : whole_1a_1_1);
                    float u1_1 = ((d1_47 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                    float2 s_47 = make_float2(max_noftz(u0_1, 0.0f), max_noftz(u1_1, 0.0f));
                    float2 act_47;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_47) : "l"(*(const unsigned long long*)&cvt_f32_49), "l"(*(const unsigned long long*)&s_47));
                    float2 _f32x2_mul_a_255 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_47, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_47), s_47), cvt_f32_2_47);
                    float2 mul_48;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_255), "l"(*(const unsigned long long*)&cvt_f32_1_47));
                    float2 mul_1_47;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_47) : "l"(*(const unsigned long long*)&act_47), "l"(*(const unsigned long long*)&cvt_f32_2_47));
                    __nv_bfloat162 bf16x2_48 = __float22bfloat162_rn(make_float2(mul_48.x, mul_48.y));
                    __nv_bfloat162 bf16x2_1_47 = __float22bfloat162_rn(make_float2(mul_1_47.x, mul_1_47.y));
                    unsigned int dgate_new_1 = __as_u32(bf16x2_48);
                    unsigned int dup_new_1 = __as_u32(bf16x2_1_47);
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
        float2 cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 cvt_f32_2_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_48 = approx_exp2((-cvt_f32_50.x) * 1.442695f);
        float e1_48 = approx_exp2((-cvt_f32_50.y) * 1.442695f);
        float d0_48 = 1.0f + e0_48;
        float d1_48 = 1.0f + e1_48;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(-d0_48), "f"(r0_48), "f"(1.0f));
        float fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_18) : "f"(r0_48), "f"(_fma_256), "f"(r0_48));
        float _fma_257;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(-d1_48), "f"(r1_48), "f"(1.0f));
        float fma_1_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_17) : "f"(r1_48), "f"(_fma_257), "f"(r1_48));
        float t0_17 = fma_18;
        float t1_17 = fma_1_17;
        band_1 = band_1 | ((fminf(t0_17, t1_17) == 0.0f) ? 1 : 0);
        float2 s_48 = make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f));
        float2 act_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_48) : "l"(*(const unsigned long long*)&cvt_f32_50), "l"(*(const unsigned long long*)&s_48));
        float2 _f32x2_mul_a_258 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_48, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), cvt_f32_2_48);
        float2 mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_258), "l"(*(const unsigned long long*)&cvt_f32_1_48));
        float2 mul_1_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_48) : "l"(*(const unsigned long long*)&act_48), "l"(*(const unsigned long long*)&cvt_f32_2_48));
        __nv_bfloat162 bf16x2_49 = __float22bfloat162_rn(make_float2(mul_49.x, mul_49.y));
        __nv_bfloat162 bf16x2_1_48 = __float22bfloat162_rn(make_float2(mul_1_48.x, mul_1_48.y));
        dgate_c[0] = __as_u32(bf16x2_49);
        dup_c[0] = __as_u32(bf16x2_1_48);
        float2 cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 cvt_f32_2_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_49 = approx_exp2((-cvt_f32_51.x) * 1.442695f);
        float e1_49 = approx_exp2((-cvt_f32_51.y) * 1.442695f);
        float d0_49 = 1.0f + e0_49;
        float d1_49 = 1.0f + e1_49;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
        float fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_19) : "f"(r0_49), "f"(_fma_259), "f"(r0_49));
        float _fma_260;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
        float fma_1_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_18) : "f"(r1_49), "f"(_fma_260), "f"(r1_49));
        float t0_18 = fma_19;
        float t1_18 = fma_1_18;
        band_1 = band_1 | ((fminf(t0_18, t1_18) == 0.0f) ? 2 : 0);
        float2 s_49 = make_float2(max_noftz(t0_18, 0.0f), max_noftz(t1_18, 0.0f));
        float2 act_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_49) : "l"(*(const unsigned long long*)&cvt_f32_51), "l"(*(const unsigned long long*)&s_49));
        float2 _f32x2_mul_a_261 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_49, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_49), s_49), cvt_f32_2_49);
        float2 mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_261), "l"(*(const unsigned long long*)&cvt_f32_1_49));
        float2 mul_1_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_49) : "l"(*(const unsigned long long*)&act_49), "l"(*(const unsigned long long*)&cvt_f32_2_49));
        __nv_bfloat162 bf16x2_50 = __float22bfloat162_rn(make_float2(mul_50.x, mul_50.y));
        __nv_bfloat162 bf16x2_1_49 = __float22bfloat162_rn(make_float2(mul_1_49.x, mul_1_49.y));
        dgate_c[1] = __as_u32(bf16x2_50);
        dup_c[1] = __as_u32(bf16x2_1_49);
        float2 cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 cvt_f32_2_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_50 = approx_exp2((-cvt_f32_52.x) * 1.442695f);
        float e1_50 = approx_exp2((-cvt_f32_52.y) * 1.442695f);
        float d0_50 = 1.0f + e0_50;
        float d1_50 = 1.0f + e1_50;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d0_50), "f"(r0_50), "f"(1.0f));
        float fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_20) : "f"(r0_50), "f"(_fma_262), "f"(r0_50));
        float _fma_263;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(-d1_50), "f"(r1_50), "f"(1.0f));
        float fma_1_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_19) : "f"(r1_50), "f"(_fma_263), "f"(r1_50));
        float t0_19 = fma_20;
        float t1_19 = fma_1_19;
        band_1 = band_1 | ((fminf(t0_19, t1_19) == 0.0f) ? 4 : 0);
        float2 s_50 = make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f));
        float2 act_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_50) : "l"(*(const unsigned long long*)&cvt_f32_52), "l"(*(const unsigned long long*)&s_50));
        float2 _f32x2_mul_a_264 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_50, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), cvt_f32_2_50);
        float2 mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_264), "l"(*(const unsigned long long*)&cvt_f32_1_50));
        float2 mul_1_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_50) : "l"(*(const unsigned long long*)&act_50), "l"(*(const unsigned long long*)&cvt_f32_2_50));
        __nv_bfloat162 bf16x2_51 = __float22bfloat162_rn(make_float2(mul_51.x, mul_51.y));
        __nv_bfloat162 bf16x2_1_50 = __float22bfloat162_rn(make_float2(mul_1_50.x, mul_1_50.y));
        dgate_c[2] = __as_u32(bf16x2_51);
        dup_c[2] = __as_u32(bf16x2_1_50);
        float2 cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 cvt_f32_2_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_51 = approx_exp2((-cvt_f32_53.x) * 1.442695f);
        float e1_51 = approx_exp2((-cvt_f32_53.y) * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_21) : "f"(r0_51), "f"(_fma_265), "f"(r0_51));
        float _fma_266;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float fma_1_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_20) : "f"(r1_51), "f"(_fma_266), "f"(r1_51));
        float t0_20 = fma_21;
        float t1_20 = fma_1_20;
        band_1 = band_1 | ((fminf(t0_20, t1_20) == 0.0f) ? 8 : 0);
        float2 s_51 = make_float2(max_noftz(t0_20, 0.0f), max_noftz(t1_20, 0.0f));
        float2 act_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_51) : "l"(*(const unsigned long long*)&cvt_f32_53), "l"(*(const unsigned long long*)&s_51));
        float2 _f32x2_mul_a_267 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_51, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_51), s_51), cvt_f32_2_51);
        float2 mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&cvt_f32_1_51));
        float2 mul_1_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_51) : "l"(*(const unsigned long long*)&act_51), "l"(*(const unsigned long long*)&cvt_f32_2_51));
        __nv_bfloat162 bf16x2_52 = __float22bfloat162_rn(make_float2(mul_52.x, mul_52.y));
        __nv_bfloat162 bf16x2_1_51 = __float22bfloat162_rn(make_float2(mul_1_51.x, mul_1_51.y));
        dgate_c[3] = __as_u32(bf16x2_52);
        dup_c[3] = __as_u32(bf16x2_1_51);
        float2 cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 cvt_f32_2_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_52 = approx_exp2((-cvt_f32_54.x) * 1.442695f);
        float e1_52 = approx_exp2((-cvt_f32_54.y) * 1.442695f);
        float d0_52 = 1.0f + e0_52;
        float d1_52 = 1.0f + e1_52;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d0_52), "f"(r0_52), "f"(1.0f));
        float fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_22) : "f"(r0_52), "f"(_fma_268), "f"(r0_52));
        float _fma_269;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(-d1_52), "f"(r1_52), "f"(1.0f));
        float fma_1_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_21) : "f"(r1_52), "f"(_fma_269), "f"(r1_52));
        float t0_21 = fma_22;
        float t1_21 = fma_1_21;
        band_1 = band_1 | ((fminf(t0_21, t1_21) == 0.0f) ? 16 : 0);
        float2 s_52 = make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f));
        float2 act_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_52) : "l"(*(const unsigned long long*)&cvt_f32_54), "l"(*(const unsigned long long*)&s_52));
        float2 _f32x2_mul_a_270 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_52, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_52), s_52), cvt_f32_2_52);
        float2 mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_270), "l"(*(const unsigned long long*)&cvt_f32_1_52));
        float2 mul_1_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_52) : "l"(*(const unsigned long long*)&act_52), "l"(*(const unsigned long long*)&cvt_f32_2_52));
        __nv_bfloat162 bf16x2_53 = __float22bfloat162_rn(make_float2(mul_53.x, mul_53.y));
        __nv_bfloat162 bf16x2_1_52 = __float22bfloat162_rn(make_float2(mul_1_52.x, mul_1_52.y));
        dgate_c[4] = __as_u32(bf16x2_53);
        dup_c[4] = __as_u32(bf16x2_1_52);
        float2 cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 cvt_f32_2_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_53 = approx_exp2((-cvt_f32_55.x) * 1.442695f);
        float e1_53 = approx_exp2((-cvt_f32_55.y) * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_23) : "f"(r0_53), "f"(_fma_271), "f"(r0_53));
        float _fma_272;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float fma_1_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_22) : "f"(r1_53), "f"(_fma_272), "f"(r1_53));
        float t0_22 = fma_23;
        float t1_22 = fma_1_22;
        band_1 = band_1 | ((fminf(t0_22, t1_22) == 0.0f) ? 32 : 0);
        float2 s_53 = make_float2(max_noftz(t0_22, 0.0f), max_noftz(t1_22, 0.0f));
        float2 act_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_53) : "l"(*(const unsigned long long*)&cvt_f32_55), "l"(*(const unsigned long long*)&s_53));
        float2 _f32x2_mul_a_273 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_53, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_53), s_53), cvt_f32_2_53);
        float2 mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_273), "l"(*(const unsigned long long*)&cvt_f32_1_53));
        float2 mul_1_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_53) : "l"(*(const unsigned long long*)&act_53), "l"(*(const unsigned long long*)&cvt_f32_2_53));
        __nv_bfloat162 bf16x2_54 = __float22bfloat162_rn(make_float2(mul_54.x, mul_54.y));
        __nv_bfloat162 bf16x2_1_53 = __float22bfloat162_rn(make_float2(mul_1_53.x, mul_1_53.y));
        dgate_c[5] = __as_u32(bf16x2_54);
        dup_c[5] = __as_u32(bf16x2_1_53);
        float2 cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 cvt_f32_2_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_54 = approx_exp2((-cvt_f32_56.x) * 1.442695f);
        float e1_54 = approx_exp2((-cvt_f32_56.y) * 1.442695f);
        float d0_54 = 1.0f + e0_54;
        float d1_54 = 1.0f + e1_54;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float _fma_274;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(-d0_54), "f"(r0_54), "f"(1.0f));
        float fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_24) : "f"(r0_54), "f"(_fma_274), "f"(r0_54));
        float _fma_275;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(-d1_54), "f"(r1_54), "f"(1.0f));
        float fma_1_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_23) : "f"(r1_54), "f"(_fma_275), "f"(r1_54));
        float t0_23 = fma_24;
        float t1_23 = fma_1_23;
        band_1 = band_1 | ((fminf(t0_23, t1_23) == 0.0f) ? 64 : 0);
        float2 s_54 = make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f));
        float2 act_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_54) : "l"(*(const unsigned long long*)&cvt_f32_56), "l"(*(const unsigned long long*)&s_54));
        float2 _f32x2_mul_a_276 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_54, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_54), s_54), cvt_f32_2_54);
        float2 mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_276), "l"(*(const unsigned long long*)&cvt_f32_1_54));
        float2 mul_1_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_54) : "l"(*(const unsigned long long*)&act_54), "l"(*(const unsigned long long*)&cvt_f32_2_54));
        __nv_bfloat162 bf16x2_55 = __float22bfloat162_rn(make_float2(mul_55.x, mul_55.y));
        __nv_bfloat162 bf16x2_1_54 = __float22bfloat162_rn(make_float2(mul_1_54.x, mul_1_54.y));
        dgate_c[6] = __as_u32(bf16x2_55);
        dup_c[6] = __as_u32(bf16x2_1_54);
        float2 cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 cvt_f32_2_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_55 = approx_exp2((-cvt_f32_57.x) * 1.442695f);
        float e1_55 = approx_exp2((-cvt_f32_57.y) * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_277;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_25) : "f"(r0_55), "f"(_fma_277), "f"(r0_55));
        float _fma_278;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float fma_1_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_24) : "f"(r1_55), "f"(_fma_278), "f"(r1_55));
        float t0_24 = fma_25;
        float t1_24 = fma_1_24;
        band_1 = band_1 | ((fminf(t0_24, t1_24) == 0.0f) ? 128 : 0);
        float2 s_55 = make_float2(max_noftz(t0_24, 0.0f), max_noftz(t1_24, 0.0f));
        float2 act_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_55) : "l"(*(const unsigned long long*)&cvt_f32_57), "l"(*(const unsigned long long*)&s_55));
        float2 _f32x2_mul_a_279 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_55, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_55), s_55), cvt_f32_2_55);
        float2 mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_279), "l"(*(const unsigned long long*)&cvt_f32_1_55));
        float2 mul_1_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_55) : "l"(*(const unsigned long long*)&act_55), "l"(*(const unsigned long long*)&cvt_f32_2_55));
        __nv_bfloat162 bf16x2_56 = __float22bfloat162_rn(make_float2(mul_56.x, mul_56.y));
        __nv_bfloat162 bf16x2_1_55 = __float22bfloat162_rn(make_float2(mul_1_55.x, mul_1_55.y));
        dgate_c[7] = __as_u32(bf16x2_56);
        dup_c[7] = __as_u32(bf16x2_1_55);
        if (band_1 != 0) {
            #pragma unroll 1
            for (int i_2 = 0; i_2 < 8; i_2++) {
                if ((band_1 >> i_2 & 1) != 0) {
                    unsigned int vec_load_5[1];
                    {
                        vec_load_5[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_4 + 2 * i_2) + 0);
                    }
                    unsigned int vec_load_1_3[1];
                    {
                        vec_load_1_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_4 + K + 2 * i_2) + 0);
                    }
                    unsigned int vec_load_2_3[1];
                    {
                        vec_load_2_3[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_c * K + col + 2 * i_2) + 0);
                    }
                    float2 cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_5[0]));
                    float2 cvt_f32_1_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_3[0]));
                    float2 cvt_f32_2_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_3[0]));
                    float e0_56 = approx_exp2((-cvt_f32_58.x) * 1.442695f);
                    float e1_56 = approx_exp2((-cvt_f32_58.y) * 1.442695f);
                    float d0_56 = 1.0f + e0_56;
                    float d1_56 = 1.0f + e1_56;
                    float r0_56 = approx_rcp(d0_56);
                    float r1_56 = approx_rcp(d1_56);
                    float _fma_280;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(-d0_56), "f"(r0_56), "f"(1.0f));
                    float fma_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_26) : "f"(r0_56), "f"(_fma_280), "f"(r0_56));
                    float _fma_281;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(-d1_56), "f"(r1_56), "f"(1.0f));
                    float fma_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_25) : "f"(r1_56), "f"(_fma_281), "f"(r1_56));
                    float t0_25 = fma_26;
                    float t1_25 = fma_1_25;
                    float d4_1a_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2) : "f"(d0_56), "f"(0.25f));
                    float r4_1a_2 = approx_rcp(d4_1a_2);
                    float _fma_283;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(-d4_1a_2), "f"(r4_1a_2), "f"(1.0f));
                    float _fma_282;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(r4_1a_2), "f"(_fma_283), "f"(r4_1a_2));
                    float big_1a_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2) : "f"(mul_f32_rn_noftz(_fma_282, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_284;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(big_1a_2), "f"(8.388608e+06f));
                    float whole_1a_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2) : "f"(_add_284), "f"(8.388608e+06f));
                    float above_1a_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2) : "f"(-d4_1a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_286;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(big_1a_2), "f"(whole_1a_2));
                    float _abs_285;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_285) : "f"(_sub_286));
                    float _add_287;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(big_1a_2), "f"(0.5f));
                    float _sub_288;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_288) : "f"(big_1a_2), "f"(0.5f));
                    float rounded_1a_2 = ((_abs_285 == 0.5f) ? ((above_1a_2 > 0.0f) ? _add_287 : _sub_288) : whole_1a_2);
                    float u0_2 = ((d0_56 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2, 7.888609e-31f), 1.7763568e-15f) : t0_25);
                    float d4_1a_2_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2_1) : "f"(d1_56), "f"(0.25f));
                    float r4_1a_2_1 = approx_rcp(d4_1a_2_1);
                    float _fma_290;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(-d4_1a_2_1), "f"(r4_1a_2_1), "f"(1.0f));
                    float _fma_289;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(r4_1a_2_1), "f"(_fma_290), "f"(r4_1a_2_1));
                    float big_1a_2_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2_1) : "f"(mul_f32_rn_noftz(_fma_289, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_291;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(big_1a_2_1), "f"(8.388608e+06f));
                    float whole_1a_2_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2_1) : "f"(_add_291), "f"(8.388608e+06f));
                    float above_1a_2_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2_1) : "f"(-d4_1a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_293;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(big_1a_2_1), "f"(whole_1a_2_1));
                    float _abs_292;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_292) : "f"(_sub_293));
                    float _add_294;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(big_1a_2_1), "f"(0.5f));
                    float _sub_295;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(big_1a_2_1), "f"(0.5f));
                    float rounded_1a_2_1 = ((_abs_292 == 0.5f) ? ((above_1a_2_1 > 0.0f) ? _add_294 : _sub_295) : whole_1a_2_1);
                    float u1_2 = ((d1_56 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_25);
                    float2 s_56 = make_float2(max_noftz(u0_2, 0.0f), max_noftz(u1_2, 0.0f));
                    float2 act_56;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_56) : "l"(*(const unsigned long long*)&cvt_f32_58), "l"(*(const unsigned long long*)&s_56));
                    float2 _f32x2_mul_a_296 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_56, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_56), s_56), cvt_f32_2_56);
                    float2 mul_57;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_296), "l"(*(const unsigned long long*)&cvt_f32_1_56));
                    float2 mul_1_56;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_56) : "l"(*(const unsigned long long*)&act_56), "l"(*(const unsigned long long*)&cvt_f32_2_56));
                    __nv_bfloat162 bf16x2_57 = __float22bfloat162_rn(make_float2(mul_57.x, mul_57.y));
                    __nv_bfloat162 bf16x2_1_56 = __float22bfloat162_rn(make_float2(mul_1_56.x, mul_1_56.y));
                    unsigned int dgate_new_2 = __as_u32(bf16x2_57);
                    unsigned int dup_new_2 = __as_u32(bf16x2_1_56);
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
        float2 cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[0]));
        float2 cvt_f32_1_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 cvt_f32_2_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[0]));
        float e0_57 = approx_exp2((-cvt_f32_59.x) * 1.442695f);
        float e1_57 = approx_exp2((-cvt_f32_59.y) * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_297;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_27) : "f"(r0_57), "f"(_fma_297), "f"(r0_57));
        float _fma_298;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float fma_1_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_26) : "f"(r1_57), "f"(_fma_298), "f"(r1_57));
        float t0_26 = fma_27;
        float t1_26 = fma_1_26;
        band_2 = band_2 | ((fminf(t0_26, t1_26) == 0.0f) ? 1 : 0);
        float2 s_57 = make_float2(max_noftz(t0_26, 0.0f), max_noftz(t1_26, 0.0f));
        float2 act_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_57) : "l"(*(const unsigned long long*)&cvt_f32_59), "l"(*(const unsigned long long*)&s_57));
        float2 _f32x2_mul_a_299 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_57, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_57), s_57), cvt_f32_2_57);
        float2 mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_299), "l"(*(const unsigned long long*)&cvt_f32_1_57));
        float2 mul_1_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_57) : "l"(*(const unsigned long long*)&act_57), "l"(*(const unsigned long long*)&cvt_f32_2_57));
        __nv_bfloat162 bf16x2_58 = __float22bfloat162_rn(make_float2(mul_58.x, mul_58.y));
        __nv_bfloat162 bf16x2_1_57 = __float22bfloat162_rn(make_float2(mul_1_57.x, mul_1_57.y));
        dgate_d[0] = __as_u32(bf16x2_58);
        dup_d[0] = __as_u32(bf16x2_1_57);
        float2 cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[1]));
        float2 cvt_f32_1_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 cvt_f32_2_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[1]));
        float e0_58 = approx_exp2((-cvt_f32_60.x) * 1.442695f);
        float e1_58 = approx_exp2((-cvt_f32_60.y) * 1.442695f);
        float d0_58 = 1.0f + e0_58;
        float d1_58 = 1.0f + e1_58;
        float r0_58 = approx_rcp(d0_58);
        float r1_58 = approx_rcp(d1_58);
        float _fma_300;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(-d0_58), "f"(r0_58), "f"(1.0f));
        float fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_28) : "f"(r0_58), "f"(_fma_300), "f"(r0_58));
        float _fma_301;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(-d1_58), "f"(r1_58), "f"(1.0f));
        float fma_1_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_27) : "f"(r1_58), "f"(_fma_301), "f"(r1_58));
        float t0_27 = fma_28;
        float t1_27 = fma_1_27;
        band_2 = band_2 | ((fminf(t0_27, t1_27) == 0.0f) ? 2 : 0);
        float2 s_58 = make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f));
        float2 act_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_58) : "l"(*(const unsigned long long*)&cvt_f32_60), "l"(*(const unsigned long long*)&s_58));
        float2 _f32x2_mul_a_302 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_58, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_58), s_58), cvt_f32_2_58);
        float2 mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_302), "l"(*(const unsigned long long*)&cvt_f32_1_58));
        float2 mul_1_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_58) : "l"(*(const unsigned long long*)&act_58), "l"(*(const unsigned long long*)&cvt_f32_2_58));
        __nv_bfloat162 bf16x2_59 = __float22bfloat162_rn(make_float2(mul_59.x, mul_59.y));
        __nv_bfloat162 bf16x2_1_58 = __float22bfloat162_rn(make_float2(mul_1_58.x, mul_1_58.y));
        dgate_d[1] = __as_u32(bf16x2_59);
        dup_d[1] = __as_u32(bf16x2_1_58);
        float2 cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[2]));
        float2 cvt_f32_1_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 cvt_f32_2_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[2]));
        float e0_59 = approx_exp2((-cvt_f32_61.x) * 1.442695f);
        float e1_59 = approx_exp2((-cvt_f32_61.y) * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_29) : "f"(r0_59), "f"(_fma_303), "f"(r0_59));
        float _fma_304;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float fma_1_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_28) : "f"(r1_59), "f"(_fma_304), "f"(r1_59));
        float t0_28 = fma_29;
        float t1_28 = fma_1_28;
        band_2 = band_2 | ((fminf(t0_28, t1_28) == 0.0f) ? 4 : 0);
        float2 s_59 = make_float2(max_noftz(t0_28, 0.0f), max_noftz(t1_28, 0.0f));
        float2 act_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_59) : "l"(*(const unsigned long long*)&cvt_f32_61), "l"(*(const unsigned long long*)&s_59));
        float2 _f32x2_mul_a_305 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_59, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_59), s_59), cvt_f32_2_59);
        float2 mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_305), "l"(*(const unsigned long long*)&cvt_f32_1_59));
        float2 mul_1_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_59) : "l"(*(const unsigned long long*)&act_59), "l"(*(const unsigned long long*)&cvt_f32_2_59));
        __nv_bfloat162 bf16x2_60 = __float22bfloat162_rn(make_float2(mul_60.x, mul_60.y));
        __nv_bfloat162 bf16x2_1_59 = __float22bfloat162_rn(make_float2(mul_1_59.x, mul_1_59.y));
        dgate_d[2] = __as_u32(bf16x2_60);
        dup_d[2] = __as_u32(bf16x2_1_59);
        float2 cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[3]));
        float2 cvt_f32_1_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 cvt_f32_2_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[3]));
        float e0_60 = approx_exp2((-cvt_f32_62.x) * 1.442695f);
        float e1_60 = approx_exp2((-cvt_f32_62.y) * 1.442695f);
        float d0_60 = 1.0f + e0_60;
        float d1_60 = 1.0f + e1_60;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float _fma_306;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(-d0_60), "f"(r0_60), "f"(1.0f));
        float fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_30) : "f"(r0_60), "f"(_fma_306), "f"(r0_60));
        float _fma_307;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(-d1_60), "f"(r1_60), "f"(1.0f));
        float fma_1_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_29) : "f"(r1_60), "f"(_fma_307), "f"(r1_60));
        float t0_29 = fma_30;
        float t1_29 = fma_1_29;
        band_2 = band_2 | ((fminf(t0_29, t1_29) == 0.0f) ? 8 : 0);
        float2 s_60 = make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f));
        float2 act_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_60) : "l"(*(const unsigned long long*)&cvt_f32_62), "l"(*(const unsigned long long*)&s_60));
        float2 _f32x2_mul_a_308 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_60, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_60), s_60), cvt_f32_2_60);
        float2 mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&cvt_f32_1_60));
        float2 mul_1_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_60) : "l"(*(const unsigned long long*)&act_60), "l"(*(const unsigned long long*)&cvt_f32_2_60));
        __nv_bfloat162 bf16x2_61 = __float22bfloat162_rn(make_float2(mul_61.x, mul_61.y));
        __nv_bfloat162 bf16x2_1_60 = __float22bfloat162_rn(make_float2(mul_1_60.x, mul_1_60.y));
        dgate_d[3] = __as_u32(bf16x2_61);
        dup_d[3] = __as_u32(bf16x2_1_60);
        float2 cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[4]));
        float2 cvt_f32_1_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 cvt_f32_2_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[4]));
        float e0_61 = approx_exp2((-cvt_f32_63.x) * 1.442695f);
        float e1_61 = approx_exp2((-cvt_f32_63.y) * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_309;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_31) : "f"(r0_61), "f"(_fma_309), "f"(r0_61));
        float _fma_310;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float fma_1_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_30) : "f"(r1_61), "f"(_fma_310), "f"(r1_61));
        float t0_30 = fma_31;
        float t1_30 = fma_1_30;
        band_2 = band_2 | ((fminf(t0_30, t1_30) == 0.0f) ? 16 : 0);
        float2 s_61 = make_float2(max_noftz(t0_30, 0.0f), max_noftz(t1_30, 0.0f));
        float2 act_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_61) : "l"(*(const unsigned long long*)&cvt_f32_63), "l"(*(const unsigned long long*)&s_61));
        float2 _f32x2_mul_a_311 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_61, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_61), s_61), cvt_f32_2_61);
        float2 mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_311), "l"(*(const unsigned long long*)&cvt_f32_1_61));
        float2 mul_1_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_61) : "l"(*(const unsigned long long*)&act_61), "l"(*(const unsigned long long*)&cvt_f32_2_61));
        __nv_bfloat162 bf16x2_62 = __float22bfloat162_rn(make_float2(mul_62.x, mul_62.y));
        __nv_bfloat162 bf16x2_1_61 = __float22bfloat162_rn(make_float2(mul_1_61.x, mul_1_61.y));
        dgate_d[4] = __as_u32(bf16x2_62);
        dup_d[4] = __as_u32(bf16x2_1_61);
        float2 cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[5]));
        float2 cvt_f32_1_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 cvt_f32_2_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[5]));
        float e0_62 = approx_exp2((-cvt_f32_64.x) * 1.442695f);
        float e1_62 = approx_exp2((-cvt_f32_64.y) * 1.442695f);
        float d0_62 = 1.0f + e0_62;
        float d1_62 = 1.0f + e1_62;
        float r0_62 = approx_rcp(d0_62);
        float r1_62 = approx_rcp(d1_62);
        float _fma_312;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(-d0_62), "f"(r0_62), "f"(1.0f));
        float fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_32) : "f"(r0_62), "f"(_fma_312), "f"(r0_62));
        float _fma_313;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(-d1_62), "f"(r1_62), "f"(1.0f));
        float fma_1_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_31) : "f"(r1_62), "f"(_fma_313), "f"(r1_62));
        float t0_31 = fma_32;
        float t1_31 = fma_1_31;
        band_2 = band_2 | ((fminf(t0_31, t1_31) == 0.0f) ? 32 : 0);
        float2 s_62 = make_float2(max_noftz(t0_31, 0.0f), max_noftz(t1_31, 0.0f));
        float2 act_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_62) : "l"(*(const unsigned long long*)&cvt_f32_64), "l"(*(const unsigned long long*)&s_62));
        float2 _f32x2_mul_a_314 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_62, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_62), s_62), cvt_f32_2_62);
        float2 mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_314), "l"(*(const unsigned long long*)&cvt_f32_1_62));
        float2 mul_1_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_62) : "l"(*(const unsigned long long*)&act_62), "l"(*(const unsigned long long*)&cvt_f32_2_62));
        __nv_bfloat162 bf16x2_63 = __float22bfloat162_rn(make_float2(mul_63.x, mul_63.y));
        __nv_bfloat162 bf16x2_1_62 = __float22bfloat162_rn(make_float2(mul_1_62.x, mul_1_62.y));
        dgate_d[5] = __as_u32(bf16x2_63);
        dup_d[5] = __as_u32(bf16x2_1_62);
        float2 cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[6]));
        float2 cvt_f32_1_63 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 cvt_f32_2_63 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[6]));
        float e0_63 = approx_exp2((-cvt_f32_65.x) * 1.442695f);
        float e1_63 = approx_exp2((-cvt_f32_65.y) * 1.442695f);
        float d0_63 = 1.0f + e0_63;
        float d1_63 = 1.0f + e1_63;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float _fma_315;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(-d0_63), "f"(r0_63), "f"(1.0f));
        float fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_33) : "f"(r0_63), "f"(_fma_315), "f"(r0_63));
        float _fma_316;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(-d1_63), "f"(r1_63), "f"(1.0f));
        float fma_1_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_32) : "f"(r1_63), "f"(_fma_316), "f"(r1_63));
        float t0_32 = fma_33;
        float t1_32 = fma_1_32;
        band_2 = band_2 | ((fminf(t0_32, t1_32) == 0.0f) ? 64 : 0);
        float2 s_63 = make_float2(max_noftz(t0_32, 0.0f), max_noftz(t1_32, 0.0f));
        float2 act_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_63) : "l"(*(const unsigned long long*)&cvt_f32_65), "l"(*(const unsigned long long*)&s_63));
        float2 _f32x2_mul_a_317 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_63, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_63), s_63), cvt_f32_2_63);
        float2 mul_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_317), "l"(*(const unsigned long long*)&cvt_f32_1_63));
        float2 mul_1_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_63) : "l"(*(const unsigned long long*)&act_63), "l"(*(const unsigned long long*)&cvt_f32_2_63));
        __nv_bfloat162 bf16x2_64 = __float22bfloat162_rn(make_float2(mul_64.x, mul_64.y));
        __nv_bfloat162 bf16x2_1_63 = __float22bfloat162_rn(make_float2(mul_1_63.x, mul_1_63.y));
        dgate_d[6] = __as_u32(bf16x2_64);
        dup_d[6] = __as_u32(bf16x2_1_63);
        float2 cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[7]));
        float2 cvt_f32_1_64 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 cvt_f32_2_64 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[7]));
        float e0_64 = approx_exp2((-cvt_f32_66.x) * 1.442695f);
        float e1_64 = approx_exp2((-cvt_f32_66.y) * 1.442695f);
        float d0_64 = 1.0f + e0_64;
        float d1_64 = 1.0f + e1_64;
        float r0_64 = approx_rcp(d0_64);
        float r1_64 = approx_rcp(d1_64);
        float _fma_318;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(-d0_64), "f"(r0_64), "f"(1.0f));
        float fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_34) : "f"(r0_64), "f"(_fma_318), "f"(r0_64));
        float _fma_319;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(-d1_64), "f"(r1_64), "f"(1.0f));
        float fma_1_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_33) : "f"(r1_64), "f"(_fma_319), "f"(r1_64));
        float t0_33 = fma_34;
        float t1_33 = fma_1_33;
        band_2 = band_2 | ((fminf(t0_33, t1_33) == 0.0f) ? 128 : 0);
        float2 s_64 = make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f));
        float2 act_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_64) : "l"(*(const unsigned long long*)&cvt_f32_66), "l"(*(const unsigned long long*)&s_64));
        float2 _f32x2_mul_a_320 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_64, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_64), s_64), cvt_f32_2_64);
        float2 mul_65;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&cvt_f32_1_64));
        float2 mul_1_64;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_64) : "l"(*(const unsigned long long*)&act_64), "l"(*(const unsigned long long*)&cvt_f32_2_64));
        __nv_bfloat162 bf16x2_65 = __float22bfloat162_rn(make_float2(mul_65.x, mul_65.y));
        __nv_bfloat162 bf16x2_1_64 = __float22bfloat162_rn(make_float2(mul_1_64.x, mul_1_64.y));
        dgate_d[7] = __as_u32(bf16x2_65);
        dup_d[7] = __as_u32(bf16x2_1_64);
        if (band_2 != 0) {
            #pragma unroll 1
            for (int i_3 = 0; i_3 < 8; i_3++) {
                if ((band_2 >> i_3 & 1) != 0) {
                    unsigned int vec_load_6[1];
                    {
                        vec_load_6[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_5 + 2 * i_3) + 0);
                    }
                    unsigned int vec_load_1_4[1];
                    {
                        vec_load_1_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_5 + K + 2 * i_3) + 0);
                    }
                    unsigned int vec_load_2_4[1];
                    {
                        vec_load_2_4[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_d * K + col + 2 * i_3) + 0);
                    }
                    float2 cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_6[0]));
                    float2 cvt_f32_1_65 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_4[0]));
                    float2 cvt_f32_2_65 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_4[0]));
                    float e0_65 = approx_exp2((-cvt_f32_67.x) * 1.442695f);
                    float e1_65 = approx_exp2((-cvt_f32_67.y) * 1.442695f);
                    float d0_65 = 1.0f + e0_65;
                    float d1_65 = 1.0f + e1_65;
                    float r0_65 = approx_rcp(d0_65);
                    float r1_65 = approx_rcp(d1_65);
                    float _fma_321;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(-d0_65), "f"(r0_65), "f"(1.0f));
                    float fma_35;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_35) : "f"(r0_65), "f"(_fma_321), "f"(r0_65));
                    float _fma_322;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(-d1_65), "f"(r1_65), "f"(1.0f));
                    float fma_1_34;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_34) : "f"(r1_65), "f"(_fma_322), "f"(r1_65));
                    float t0_34 = fma_35;
                    float t1_34 = fma_1_34;
                    float d4_1a_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3) : "f"(d0_65), "f"(0.25f));
                    float r4_1a_3 = approx_rcp(d4_1a_3);
                    float _fma_324;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(-d4_1a_3), "f"(r4_1a_3), "f"(1.0f));
                    float _fma_323;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(r4_1a_3), "f"(_fma_324), "f"(r4_1a_3));
                    float big_1a_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3) : "f"(mul_f32_rn_noftz(_fma_323, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_325;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(big_1a_3), "f"(8.388608e+06f));
                    float whole_1a_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3) : "f"(_add_325), "f"(8.388608e+06f));
                    float above_1a_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3) : "f"(-d4_1a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_327;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(big_1a_3), "f"(whole_1a_3));
                    float _abs_326;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_326) : "f"(_sub_327));
                    float _add_328;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(big_1a_3), "f"(0.5f));
                    float _sub_329;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(big_1a_3), "f"(0.5f));
                    float rounded_1a_3 = ((_abs_326 == 0.5f) ? ((above_1a_3 > 0.0f) ? _add_328 : _sub_329) : whole_1a_3);
                    float u0_3 = ((d0_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3, 7.888609e-31f), 1.7763568e-15f) : t0_34);
                    float d4_1a_3_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3_1) : "f"(d1_65), "f"(0.25f));
                    float r4_1a_3_1 = approx_rcp(d4_1a_3_1);
                    float _fma_331;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(-d4_1a_3_1), "f"(r4_1a_3_1), "f"(1.0f));
                    float _fma_330;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(r4_1a_3_1), "f"(_fma_331), "f"(r4_1a_3_1));
                    float big_1a_3_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3_1) : "f"(mul_f32_rn_noftz(_fma_330, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_332;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(big_1a_3_1), "f"(8.388608e+06f));
                    float whole_1a_3_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3_1) : "f"(_add_332), "f"(8.388608e+06f));
                    float above_1a_3_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3_1) : "f"(-d4_1a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_334;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(big_1a_3_1), "f"(whole_1a_3_1));
                    float _abs_333;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_333) : "f"(_sub_334));
                    float _add_335;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(big_1a_3_1), "f"(0.5f));
                    float _sub_336;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(big_1a_3_1), "f"(0.5f));
                    float rounded_1a_3_1 = ((_abs_333 == 0.5f) ? ((above_1a_3_1 > 0.0f) ? _add_335 : _sub_336) : whole_1a_3_1);
                    float u1_3 = ((d1_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_34);
                    float2 s_65 = make_float2(max_noftz(u0_3, 0.0f), max_noftz(u1_3, 0.0f));
                    float2 act_65;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_65) : "l"(*(const unsigned long long*)&cvt_f32_67), "l"(*(const unsigned long long*)&s_65));
                    float2 _f32x2_mul_a_337 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_65, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_65), s_65), cvt_f32_2_65);
                    float2 mul_66;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_337), "l"(*(const unsigned long long*)&cvt_f32_1_65));
                    float2 mul_1_65;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_65) : "l"(*(const unsigned long long*)&act_65), "l"(*(const unsigned long long*)&cvt_f32_2_65));
                    __nv_bfloat162 bf16x2_66 = __float22bfloat162_rn(make_float2(mul_66.x, mul_66.y));
                    __nv_bfloat162 bf16x2_1_65 = __float22bfloat162_rn(make_float2(mul_1_65.x, mul_1_65.y));
                    unsigned int dgate_new_3 = __as_u32(bf16x2_66);
                    unsigned int dup_new_3 = __as_u32(bf16x2_1_65);
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_0 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_0 + 3072)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src0_1_1 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src0_1_1 + 3072)
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_403;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_403) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_402;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_402) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_403));
    words_a[0] = _mov_b32_pack_u16x2_402;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[16]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[17]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_405;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_405) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_404;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_404) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_405));
    words_b[0] = _mov_b32_pack_u16x2_404;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_407;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_406;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_406) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_407));
    words_a[1] = _mov_b32_pack_u16x2_406;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(cols[18]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[19]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_409;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_408;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_408) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_409));
    words_b[1] = _mov_b32_pack_u16x2_408;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_411;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_410;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_410) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_411));
    words_a[2] = _mov_b32_pack_u16x2_410;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(cols[20]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols[21]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_413;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_412;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_412) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_413));
    words_b[2] = _mov_b32_pack_u16x2_412;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_415;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_414;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_414) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_415));
    words_a[3] = _mov_b32_pack_u16x2_414;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols[22]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols[23]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_417;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_416;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_416) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_417));
    words_b[3] = _mov_b32_pack_u16x2_416;
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(cols[8]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(cols[9]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_419;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_418;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_418) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_419));
    words_a[4] = _mov_b32_pack_u16x2_418;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(cols[24]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(cols[25]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_421;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_420;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_420) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_421));
    words_b[4] = _mov_b32_pack_u16x2_420;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(cols[10]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(cols[11]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_423;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_422;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_422) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_423));
    words_a[5] = _mov_b32_pack_u16x2_422;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(cols[26]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(cols[27]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_425;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_425) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_424;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_424) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_425));
    words_b[5] = _mov_b32_pack_u16x2_424;
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(cols[12]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(cols[13]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_427;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_427) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_426;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_426) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_427));
    words_a[6] = _mov_b32_pack_u16x2_426;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(cols[28]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(cols[29]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_429;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_429) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_428;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_428) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_429));
    words_b[6] = _mov_b32_pack_u16x2_428;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(cols[14]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(cols[15]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_431;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_431) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_431) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_430;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_430) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_431));
    words_a[7] = _mov_b32_pack_u16x2_430;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(cols[30]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(cols[31]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_433;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_433) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_433) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_432;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_432) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_433));
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
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_0);
    int lane_8 = tid & 31;
    int warp_9 = tid >> 5;
    int k8_10 = lane_8 & 7;
    int srow_11 = (k8_10 >> 1) * 32 + (lane_8 >> 3) * 2 + (k8_10 & 1);
    unsigned int sswizzle_12 = (unsigned int)(srow_11 >> 4 & 6 | srow_11 & 1);
    unsigned int cols_13[32];
    unsigned int strip_chunk0_0 = (unsigned int)(warp_9 * 2);
    unsigned int strip_src0_0 = tiles_addr + 16384 + (unsigned int)(srow_11 * 128) + ((strip_chunk0_0 ^ sswizzle_12) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[0]), "=r"(cols_13[1]), "=r"(cols_13[2]), "=r"(cols_13[3])
        : "r"(strip_src0_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[4]), "=r"(cols_13[5]), "=r"(cols_13[6]), "=r"(cols_13[7])
        : "r"(strip_src0_0 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[8]), "=r"(cols_13[9]), "=r"(cols_13[10]), "=r"(cols_13[11])
        : "r"(strip_src0_0 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[12]), "=r"(cols_13[13]), "=r"(cols_13[14]), "=r"(cols_13[15])
        : "r"(strip_src0_0 + 3072)
        : "memory");
    unsigned int strip_chunk0_1 = (unsigned int)(warp_9 * 2 + 1);
    unsigned int strip_src0_1 = tiles_addr + 16384 + (unsigned int)(srow_11 * 128) + ((strip_chunk0_1 ^ sswizzle_12) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[16]), "=r"(cols_13[17]), "=r"(cols_13[18]), "=r"(cols_13[19])
        : "r"(strip_src0_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[20]), "=r"(cols_13[21]), "=r"(cols_13[22]), "=r"(cols_13[23])
        : "r"(strip_src0_1 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[24]), "=r"(cols_13[25]), "=r"(cols_13[26]), "=r"(cols_13[27])
        : "r"(strip_src0_1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_13[28]), "=r"(cols_13[29]), "=r"(cols_13[30]), "=r"(cols_13[31])
        : "r"(strip_src0_1 + 3072)
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
    unsigned int amax_bits_1 = (col_max_bits_25 & 65535) << 16;
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
    unsigned int amax_bits_2 = (col_max_bits_25 >> 16 & 65535) << 16;
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
    int gcol_a_36 = bx * 64 + K + warp_9 * 16 + (lane_8 >> 2);
    int gcol_b_37 = gcol_a_36 + 8;
    int out_row_38 = by * 128 + (lane_8 & 3) * 32;
    unsigned int words_a_39[8];
    unsigned int words_b_40[8];
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(cols_13[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(cols_13[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_501;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_500;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_500) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_501));
    words_a_39[0] = _mov_b32_pack_u16x2_500;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(cols_13[16]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols_13[17]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_503;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_502;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_502) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_503));
    words_b_40[0] = _mov_b32_pack_u16x2_502;
    uint32_t bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(cols_13[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(cols_13[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_505;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_504;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_504) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_505));
    words_a_39[1] = _mov_b32_pack_u16x2_504;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols_13[18]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols_13[19]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_507;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_506;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_506) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_507));
    words_b_40[1] = _mov_b32_pack_u16x2_506;
    uint32_t bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(cols_13[4]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(cols_13[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_509;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_509) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_508;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_508) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_509));
    words_a_39[2] = _mov_b32_pack_u16x2_508;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols_13[20]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols_13[21]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_511;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_510;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_510) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_511));
    words_b_40[2] = _mov_b32_pack_u16x2_510;
    uint32_t bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(cols_13[6]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(cols_13[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_513;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_512;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_512) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_513));
    words_a_39[3] = _mov_b32_pack_u16x2_512;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols_13[22]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols_13[23]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_515;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_515) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_514;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_514) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_515));
    words_b_40[3] = _mov_b32_pack_u16x2_514;
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols_13[8]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols_13[9]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_517;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_516;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_516) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_517));
    words_a_39[4] = _mov_b32_pack_u16x2_516;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols_13[24]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols_13[25]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_519;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_519) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_518;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_518) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_519));
    words_b_40[4] = _mov_b32_pack_u16x2_518;
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols_13[10]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols_13[11]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_521;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_521) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_520;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_520) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_521));
    words_a_39[5] = _mov_b32_pack_u16x2_520;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols_13[26]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols_13[27]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_523;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_523) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_522;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_522) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_523));
    words_b_40[5] = _mov_b32_pack_u16x2_522;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols_13[12]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols_13[13]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_525;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_525) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_524;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_524) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_525));
    words_a_39[6] = _mov_b32_pack_u16x2_524;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols_13[28]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols_13[29]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_527;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_527) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_527) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_526;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_526) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_527));
    words_b_40[6] = _mov_b32_pack_u16x2_526;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols_13[14]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols_13[15]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_529;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_529) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_529) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_528;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_528) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_529));
    words_a_39[7] = _mov_b32_pack_u16x2_528;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols_13[30]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols_13[31]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_531;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_531) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_531) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_530;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_530) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_531));
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
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_43) + (0)) = (unsigned char)(scale_1);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_44) + (0)) = (unsigned char)(scale_2);
}

} // extern "C"
