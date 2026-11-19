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

#define QUANT_UNITS 1
#define SF_TILE_BYTES 512

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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_fp8_projection_25221ffed4cdc1b2d4d7(__nv_bfloat16* __restrict__ x, uint8_t* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int K, int units_per_row, int sf_k_sets, int sf_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int half = lane >> 4;
    int half_lane = lane & 15;
    int unit = (bid * 4 + warp) * 2 + half;
    int row = unit / units_per_row;
    int row_c = ((row) < (M - 1) ? (row) : (M - 1));
    int k_block0 = (unit - row * units_per_row) * QUANT_UNITS;
    unsigned long long row_base = (unsigned long long)row_c * (unsigned long long)K;
    int m_tile = row_c / sf_rows;
    int row_in_tile = row_c - m_tile * sf_rows;
    unsigned long long sf_row_base = (unsigned long long)m_tile * (unsigned long long)sf_k_sets * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(row_in_tile & 31) * 16ull + (unsigned long long)(row_in_tile >> 5) * 4ull;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int staged[4];
    #pragma unroll
    for (int u = 0; u < QUANT_UNITS; u++) {
        int k_ld = (k_block0 + u) * 128 + half_lane * 8;
        {
            uint32_t _uv4_0_0;
            uint32_t _uv4_0_1;
            uint32_t _uv4_0_2;
            uint32_t _uv4_0_3;
            asm volatile("ld.global.nc.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(x + (row_base + (unsigned long long)k_ld) + (0))), "l"(0x12F0000000000000ULL) : "memory");
            staged[4 * u + 0] = _uv4_0_0;
            staged[4 * u + 1] = _uv4_0_1;
            staged[4 * u + 2] = _uv4_0_2;
            staged[4 * u + 3] = _uv4_0_3;
        }
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < QUANT_UNITS; u_1++) {
        int k_block = k_block0 + u_1;
        int k = k_block * 128 + half_lane * 8;
        uint32_t abs0;
        asm("abs.bf16x2 %0, %1;" : "=r"(abs0) : "r"(staged[4 * u_1]));
        uint32_t abs1;
        asm("abs.bf16x2 %0, %1;" : "=r"(abs1) : "r"(staged[4 * u_1 + 1]));
        uint32_t abs2;
        asm("abs.bf16x2 %0, %1;" : "=r"(abs2) : "r"(staged[4 * u_1 + 2]));
        uint32_t abs3;
        asm("abs.bf16x2 %0, %1;" : "=r"(abs3) : "r"(staged[4 * u_1 + 3]));
        uint32_t low_max;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(low_max) : "r"(abs0), "r"(abs1));
        uint32_t high_max;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(high_max) : "r"(abs2), "r"(abs3));
        uint32_t pair_max;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(pair_max) : "r"(low_max), "r"(high_max));
        uint32_t unit_max;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(unit_max) : "r"(pair_max), "r"(pair_max >> 16));
        float absmax = __uint_as_float(unit_max << 16);
        float other_0 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
        absmax = fmaxf(absmax, other_0);
        float other_1 = __shfl_xor_sync(0xFFFFFFFF, absmax, 2);
        absmax = fmaxf(absmax, other_1);
        float other_2 = __shfl_xor_sync(0xFFFFFFFF, absmax, 4);
        absmax = fmaxf(absmax, other_2);
        float other_3 = __shfl_xor_sync(0xFFFFFFFF, absmax, 8);
        absmax = fmaxf(absmax, other_3);
        absmax = fmaxf(absmax, 1e-04f);
        float scale = absmax / 448.0f;
        unsigned int scale_bits = __as_u32(scale);
        unsigned int exponent = scale_bits >> 23 & 255;
        unsigned int mantissa = scale_bits & 8388607;
        unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
        unsigned int _max_1 = ((exponent + has_mantissa) > (1) ? (exponent + has_mantissa) : (1));
        unsigned int scale_byte = ((_max_1) < (254) ? (_max_1) : (254));
        unsigned int inverse_pair = (254 - scale_byte << 7) * 65537;
        unsigned int codes[2];
        uint32_t low_pair_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(low_pair_0) : "r"(staged[4 * u_1]), "r"(inverse_pair));
        uint32_t high_pair_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(high_pair_0) : "r"(staged[4 * u_1 + 1]), "r"(inverse_pair));
        uint16_t _cvt_satfinite_e4m3x2_bf16x2_3;
        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "r"(low_pair_0));
        #else
        {
        const uint32_t _bf16x2 = (uint32_t)(low_pair_0);
        const float _lo = __uint_as_float(_bf16x2 << 16);
        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
        }
        #endif
        uint16_t _cvt_satfinite_e4m3x2_bf16x2_4;
        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_4) : "r"(high_pair_0));
        #else
        {
        const uint32_t _bf16x2 = (uint32_t)(high_pair_0);
        const float _lo = __uint_as_float(_bf16x2 << 16);
        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
        }
        #endif
        uint32_t _mov_b32_pack_u16x2_2;
        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_2) : "h"(_cvt_satfinite_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_4));
        codes[0] = _mov_b32_pack_u16x2_2;
        uint32_t low_pair_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(low_pair_1) : "r"(staged[4 * u_1 + 2]), "r"(inverse_pair));
        uint32_t high_pair_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(high_pair_1) : "r"(staged[4 * u_1 + 2 + 1]), "r"(inverse_pair));
        uint16_t _cvt_satfinite_e4m3x2_bf16x2_6;
        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_6) : "r"(low_pair_1));
        #else
        {
        const uint32_t _bf16x2 = (uint32_t)(low_pair_1);
        const float _lo = __uint_as_float(_bf16x2 << 16);
        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
        }
        #endif
        uint16_t _cvt_satfinite_e4m3x2_bf16x2_7;
        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "r"(high_pair_1));
        #else
        {
        const uint32_t _bf16x2 = (uint32_t)(high_pair_1);
        const float _lo = __uint_as_float(_bf16x2 << 16);
        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
        }
        #endif
        uint32_t _mov_b32_pack_u16x2_5;
        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_5) : "h"(_cvt_satfinite_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_7));
        codes[1] = _mov_b32_pack_u16x2_5;
        if (row < M) {
            asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1, %2}, %3;" :: "l"(a_q + (row_base + (unsigned long long)k)), "r"(codes[(0) + 0]), "r"(codes[(0) + 1]), "l"(0x14F0000000000000ULL) : "memory");
            if (half_lane == 0) {
                unsigned long long sf_addr = sf_row_base + (unsigned long long)k_block * (unsigned long long)SF_TILE_BYTES;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    asm volatile("st.global.L2::cache_hint.b8 [%0], %1, %2;" :: "l"(a_sf + (sf_addr + (unsigned long long)c)), "r"((unsigned int)(unsigned char)(scale_byte)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
}

} // extern "C"
