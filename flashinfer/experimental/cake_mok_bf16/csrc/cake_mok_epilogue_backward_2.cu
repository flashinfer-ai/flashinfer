/*
 * Copyright 2026 Cursor Research
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from the Apache-2.0 Mixture of Kittens BF16 training kernels.
 * Modified: generated CUDA implementation and standalone TVM-FFI bindings.
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

#include <math_constants.h>


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
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mok_epilogue_backward_2(__nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ routed, float* __restrict__ scores, unsigned int* __restrict__ output, int tokens, int hidden, int ctas)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int* shared_words = reinterpret_cast<unsigned int*>(shared);
    unsigned int* routed_words = reinterpret_cast<unsigned int*>(routed);
    int passes = (hidden + 2047) / 2048;
    #pragma unroll 1
    for (int token = bid; token < tokens; token += ctas) {
        unsigned long long row = (unsigned long long)token * (unsigned long long)hidden;
        #pragma unroll 1
        for (int p = 0; p < passes; p++) {
            int col = p * 2048 + tid * 8;
            if (col < hidden) {
                unsigned int values[12];
                {
                    uint32_t _uv4_0_0;
                    uint32_t _uv4_0_1;
                    uint32_t _uv4_0_2;
                    uint32_t _uv4_0_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(shared_words + (row + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[0 + 0] = _uv4_0_0;
                    values[0 + 1] = _uv4_0_1;
                    values[0 + 2] = _uv4_0_2;
                    values[0 + 3] = _uv4_0_3;
                }
                {
                    uint32_t _uv4_1_0;
                    uint32_t _uv4_1_1;
                    uint32_t _uv4_1_2;
                    uint32_t _uv4_1_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(routed_words + (row * 2ull + 0ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[4 + 0] = _uv4_1_0;
                    values[4 + 1] = _uv4_1_1;
                    values[4 + 2] = _uv4_1_2;
                    values[4 + 3] = _uv4_1_3;
                }
                {
                    uint32_t _uv4_2_0;
                    uint32_t _uv4_2_1;
                    uint32_t _uv4_2_2;
                    uint32_t _uv4_2_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(routed_words + (row * 2ull + 1ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[8 + 0] = _uv4_2_0;
                    values[8 + 1] = _uv4_2_1;
                    values[8 + 2] = _uv4_2_2;
                    values[8 + 3] = _uv4_2_3;
                }
                float acc[8];
                float2 first_0 = cake_bf16x2_to_f32x2(__as_bf16x2(values[0]));
                acc[0] = first_0.x;
                acc[1] = first_0.y;
                float2 first_1 = cake_bf16x2_to_f32x2(__as_bf16x2(values[1]));
                acc[2] = first_1.x;
                acc[3] = first_1.y;
                float2 first_2 = cake_bf16x2_to_f32x2(__as_bf16x2(values[2]));
                acc[4] = first_2.x;
                acc[5] = first_2.y;
                float2 first_3 = cake_bf16x2_to_f32x2(__as_bf16x2(values[3]));
                acc[6] = first_3.x;
                acc[7] = first_3.y;
                float2 term_0 = cake_bf16x2_to_f32x2(__as_bf16x2(values[4]));
                acc[0] = acc[0] + term_0.x;
                acc[1] = acc[1] + term_0.y;
                float2 term_1 = cake_bf16x2_to_f32x2(__as_bf16x2(values[5]));
                acc[2] = acc[2] + term_1.x;
                acc[3] = acc[3] + term_1.y;
                float2 term_2 = cake_bf16x2_to_f32x2(__as_bf16x2(values[6]));
                acc[4] = acc[4] + term_2.x;
                acc[5] = acc[5] + term_2.y;
                float2 term_3 = cake_bf16x2_to_f32x2(__as_bf16x2(values[7]));
                acc[6] = acc[6] + term_3.x;
                acc[7] = acc[7] + term_3.y;
                float2 term_4 = cake_bf16x2_to_f32x2(__as_bf16x2(values[8]));
                acc[0] = acc[0] + term_4.x;
                acc[1] = acc[1] + term_4.y;
                float2 term_5 = cake_bf16x2_to_f32x2(__as_bf16x2(values[9]));
                acc[2] = acc[2] + term_5.x;
                acc[3] = acc[3] + term_5.y;
                float2 term_6 = cake_bf16x2_to_f32x2(__as_bf16x2(values[10]));
                acc[4] = acc[4] + term_6.x;
                acc[5] = acc[5] + term_6.y;
                float2 term_7 = cake_bf16x2_to_f32x2(__as_bf16x2(values[11]));
                acc[6] = acc[6] + term_7.x;
                acc[7] = acc[7] + term_7.y;
                unsigned int packed[4];
                packed[0] = __as_u32(__float22bfloat162_rn(make_float2(acc[0], acc[1])));
                packed[1] = __as_u32(__float22bfloat162_rn(make_float2(acc[2], acc[3])));
                packed[2] = __as_u32(__float22bfloat162_rn(make_float2(acc[4], acc[5])));
                packed[3] = __as_u32(__float22bfloat162_rn(make_float2(acc[6], acc[7])));
                reinterpret_cast<int4*>(output + (row + (unsigned long long)col) / 2ull)[0] = reinterpret_cast<int4*>(packed)[0];
            }
        }
    }
}

} // extern "C"
