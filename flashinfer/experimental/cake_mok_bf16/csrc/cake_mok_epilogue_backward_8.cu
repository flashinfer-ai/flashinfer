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
kernel_cake_mok_epilogue_backward_8(__nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ routed, float* __restrict__ scores, unsigned int* __restrict__ output, int tokens, int hidden, int ctas)
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
                unsigned int values[36];
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
                        : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(routed_words + (row * 8ull + 0ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
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
                        : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(routed_words + (row * 8ull + 1ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[8 + 0] = _uv4_2_0;
                    values[8 + 1] = _uv4_2_1;
                    values[8 + 2] = _uv4_2_2;
                    values[8 + 3] = _uv4_2_3;
                }
                {
                    uint32_t _uv4_3_0;
                    uint32_t _uv4_3_1;
                    uint32_t _uv4_3_2;
                    uint32_t _uv4_3_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(routed_words + (row * 8ull + 2ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[12 + 0] = _uv4_3_0;
                    values[12 + 1] = _uv4_3_1;
                    values[12 + 2] = _uv4_3_2;
                    values[12 + 3] = _uv4_3_3;
                }
                {
                    uint32_t _uv4_4_0;
                    uint32_t _uv4_4_1;
                    uint32_t _uv4_4_2;
                    uint32_t _uv4_4_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_4_0), "=r"(_uv4_4_1), "=r"(_uv4_4_2), "=r"(_uv4_4_3) : "l"((const void*)(routed_words + (row * 8ull + 3ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[16 + 0] = _uv4_4_0;
                    values[16 + 1] = _uv4_4_1;
                    values[16 + 2] = _uv4_4_2;
                    values[16 + 3] = _uv4_4_3;
                }
                {
                    uint32_t _uv4_5_0;
                    uint32_t _uv4_5_1;
                    uint32_t _uv4_5_2;
                    uint32_t _uv4_5_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(routed_words + (row * 8ull + 4ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[20 + 0] = _uv4_5_0;
                    values[20 + 1] = _uv4_5_1;
                    values[20 + 2] = _uv4_5_2;
                    values[20 + 3] = _uv4_5_3;
                }
                {
                    uint32_t _uv4_6_0;
                    uint32_t _uv4_6_1;
                    uint32_t _uv4_6_2;
                    uint32_t _uv4_6_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_6_0), "=r"(_uv4_6_1), "=r"(_uv4_6_2), "=r"(_uv4_6_3) : "l"((const void*)(routed_words + (row * 8ull + 5ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[24 + 0] = _uv4_6_0;
                    values[24 + 1] = _uv4_6_1;
                    values[24 + 2] = _uv4_6_2;
                    values[24 + 3] = _uv4_6_3;
                }
                {
                    uint32_t _uv4_7_0;
                    uint32_t _uv4_7_1;
                    uint32_t _uv4_7_2;
                    uint32_t _uv4_7_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(routed_words + (row * 8ull + 6ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[28 + 0] = _uv4_7_0;
                    values[28 + 1] = _uv4_7_1;
                    values[28 + 2] = _uv4_7_2;
                    values[28 + 3] = _uv4_7_3;
                }
                {
                    uint32_t _uv4_8_0;
                    uint32_t _uv4_8_1;
                    uint32_t _uv4_8_2;
                    uint32_t _uv4_8_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_8_0), "=r"(_uv4_8_1), "=r"(_uv4_8_2), "=r"(_uv4_8_3) : "l"((const void*)(routed_words + (row * 8ull + 7ull * (unsigned long long)hidden + (unsigned long long)col) / 2ull + (0))) : "memory");
                    values[32 + 0] = _uv4_8_0;
                    values[32 + 1] = _uv4_8_1;
                    values[32 + 2] = _uv4_8_2;
                    values[32 + 3] = _uv4_8_3;
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
                float2 term_8 = cake_bf16x2_to_f32x2(__as_bf16x2(values[12]));
                acc[0] = acc[0] + term_8.x;
                acc[1] = acc[1] + term_8.y;
                float2 term_9 = cake_bf16x2_to_f32x2(__as_bf16x2(values[13]));
                acc[2] = acc[2] + term_9.x;
                acc[3] = acc[3] + term_9.y;
                float2 term_10 = cake_bf16x2_to_f32x2(__as_bf16x2(values[14]));
                acc[4] = acc[4] + term_10.x;
                acc[5] = acc[5] + term_10.y;
                float2 term_11 = cake_bf16x2_to_f32x2(__as_bf16x2(values[15]));
                acc[6] = acc[6] + term_11.x;
                acc[7] = acc[7] + term_11.y;
                float2 term_12 = cake_bf16x2_to_f32x2(__as_bf16x2(values[16]));
                acc[0] = acc[0] + term_12.x;
                acc[1] = acc[1] + term_12.y;
                float2 term_13 = cake_bf16x2_to_f32x2(__as_bf16x2(values[17]));
                acc[2] = acc[2] + term_13.x;
                acc[3] = acc[3] + term_13.y;
                float2 term_14 = cake_bf16x2_to_f32x2(__as_bf16x2(values[18]));
                acc[4] = acc[4] + term_14.x;
                acc[5] = acc[5] + term_14.y;
                float2 term_15 = cake_bf16x2_to_f32x2(__as_bf16x2(values[19]));
                acc[6] = acc[6] + term_15.x;
                acc[7] = acc[7] + term_15.y;
                float2 term_16 = cake_bf16x2_to_f32x2(__as_bf16x2(values[20]));
                acc[0] = acc[0] + term_16.x;
                acc[1] = acc[1] + term_16.y;
                float2 term_17 = cake_bf16x2_to_f32x2(__as_bf16x2(values[21]));
                acc[2] = acc[2] + term_17.x;
                acc[3] = acc[3] + term_17.y;
                float2 term_18 = cake_bf16x2_to_f32x2(__as_bf16x2(values[22]));
                acc[4] = acc[4] + term_18.x;
                acc[5] = acc[5] + term_18.y;
                float2 term_19 = cake_bf16x2_to_f32x2(__as_bf16x2(values[23]));
                acc[6] = acc[6] + term_19.x;
                acc[7] = acc[7] + term_19.y;
                float2 term_20 = cake_bf16x2_to_f32x2(__as_bf16x2(values[24]));
                acc[0] = acc[0] + term_20.x;
                acc[1] = acc[1] + term_20.y;
                float2 term_21 = cake_bf16x2_to_f32x2(__as_bf16x2(values[25]));
                acc[2] = acc[2] + term_21.x;
                acc[3] = acc[3] + term_21.y;
                float2 term_22 = cake_bf16x2_to_f32x2(__as_bf16x2(values[26]));
                acc[4] = acc[4] + term_22.x;
                acc[5] = acc[5] + term_22.y;
                float2 term_23 = cake_bf16x2_to_f32x2(__as_bf16x2(values[27]));
                acc[6] = acc[6] + term_23.x;
                acc[7] = acc[7] + term_23.y;
                float2 term_24 = cake_bf16x2_to_f32x2(__as_bf16x2(values[28]));
                acc[0] = acc[0] + term_24.x;
                acc[1] = acc[1] + term_24.y;
                float2 term_25 = cake_bf16x2_to_f32x2(__as_bf16x2(values[29]));
                acc[2] = acc[2] + term_25.x;
                acc[3] = acc[3] + term_25.y;
                float2 term_26 = cake_bf16x2_to_f32x2(__as_bf16x2(values[30]));
                acc[4] = acc[4] + term_26.x;
                acc[5] = acc[5] + term_26.y;
                float2 term_27 = cake_bf16x2_to_f32x2(__as_bf16x2(values[31]));
                acc[6] = acc[6] + term_27.x;
                acc[7] = acc[7] + term_27.y;
                float2 term_28 = cake_bf16x2_to_f32x2(__as_bf16x2(values[32]));
                acc[0] = acc[0] + term_28.x;
                acc[1] = acc[1] + term_28.y;
                float2 term_29 = cake_bf16x2_to_f32x2(__as_bf16x2(values[33]));
                acc[2] = acc[2] + term_29.x;
                acc[3] = acc[3] + term_29.y;
                float2 term_30 = cake_bf16x2_to_f32x2(__as_bf16x2(values[34]));
                acc[4] = acc[4] + term_30.x;
                acc[5] = acc[5] + term_30.y;
                float2 term_31 = cake_bf16x2_to_f32x2(__as_bf16x2(values[35]));
                acc[6] = acc[6] + term_31.x;
                acc[7] = acc[7] + term_31.y;
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
