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

// This unit serves two programs: cake_grouped_mxfp8_quantize_0b198af0ce3fd48455dc (CAKE_PROGRAM_VARIANT 0, the default) and
// cake_grouped_mxfp8_quantize_10880bc62476fde7de46 (CAKE_PROGRAM_VARIANT 1).  CAKE_PROGRAM_KERNEL names the kernel it defines.  A build
// selects the second program with the compile definitions
// -DCAKE_PROGRAM_VARIANT=1 -DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_10880bc62476fde7de46.
#ifndef CAKE_PROGRAM_VARIANT
#define CAKE_PROGRAM_VARIANT 0
#endif
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_grouped_mxfp8_quantize_0b198af0ce3fd48455dc
#endif

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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)
#if CAKE_PROGRAM_VARIANT == 1


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
#else
#endif

extern "C" {

__global__ __launch_bounds__(128) void
CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ x, int* __restrict__ mask, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales, int M, int K, int PADDED_K, int PM_TILES, int PK_TILES, CakeFastDivmod BLOCKS_PER_ROW, unsigned long long TOTAL_TASKS, unsigned int ROWS, unsigned int SUBROWS)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int group = blockIdx.z;
    int _vec_load_0[1];
    {
        _vec_load_0[0] = *reinterpret_cast<const int*>(mask + group);
    }
    unsigned int valid_rows = (((unsigned int)_vec_load_0[0]) < ((unsigned int)M) ? ((unsigned int)_vec_load_0[0]) : ((unsigned int)M));
    unsigned int tile = blockIdx.y;
    if (tile * (ROWS * SUBROWS) < valid_rows) {
        if (tid == 0) {
            if (ROWS > 1 && valid_rows < (unsigned int)M) {
                unsigned int k_lo = (unsigned int)blockIdx.x * 4096;
                unsigned int cta_span = (((unsigned int)K - k_lo) < (4096) ? ((unsigned int)K - k_lo) : (4096));
                unsigned int first = tile * (ROWS * SUBROWS);
                unsigned int next_first = first + SUBROWS;
                if (next_first < valid_rows) {
                    unsigned int next_rows = ((valid_rows - next_first) < (SUBROWS) ? (valid_rows - next_first) : (SUBROWS));
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (((unsigned long long)group * (unsigned long long)M + (unsigned long long)next_first) * (unsigned long long)K + (unsigned long long)k_lo)), "r"((uint32_t)(((next_rows - 1) * (unsigned int)K + cta_span) * 2)) : "memory");
                }
                unsigned int ahead = group + 3;
                if (ahead < (unsigned int)gridDim.z) {
                    unsigned int sparse_rows = (unsigned int)((unsigned long long)M * 3 / 12);
                    int _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const int*>(mask + ahead);
                    }
                    unsigned int ahead_valid = (((unsigned int)_vec_load_2[0]) < ((unsigned int)M) ? ((unsigned int)_vec_load_2[0]) : ((unsigned int)M));
                    if (first < ahead_valid && ahead_valid <= sparse_rows) {
                        unsigned int ahead_rows = ((ahead_valid - first) < (ROWS * SUBROWS) ? (ahead_valid - first) : (ROWS * SUBROWS));
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (((unsigned long long)ahead * (unsigned long long)M + (unsigned long long)first) * (unsigned long long)K + (unsigned long long)k_lo)), "r"((uint32_t)(((ahead_rows - 1) * (unsigned int)K + cta_span) * 2)) : "memory");
                    }
                }
            }
        }
        unsigned int tid_0 = (unsigned int)tid;
        unsigned int flat = (unsigned int)blockIdx.x * 128 + tid_0;
        uint32_t _fast_div_q_0 = (BLOCKS_PER_ROW.divisor != 1) ? (__umulhi((uint32_t)(flat), BLOCKS_PER_ROW.multiplier) >> BLOCKS_PER_ROW.shift_right) : (uint32_t)(flat);
        uint32_t _fast_div_r_0 = (uint32_t)(flat) - _fast_div_q_0 * (uint32_t)(BLOCKS_PER_ROW.divisor);
        if (_fast_div_q_0 < SUBROWS) {
            unsigned int first_row = tile * (ROWS * SUBROWS) + _fast_div_q_0;
            if (first_row < valid_rows) {
                unsigned int k0 = _fast_div_r_0 * 32;
                unsigned int row = first_row;
                unsigned long long global_row = (unsigned long long)group * (unsigned long long)M + (unsigned long long)row;
                unsigned long long x_row = global_row * (unsigned long long)K;
                unsigned long long q_row = global_row * (unsigned long long)((uint32_t)(BLOCKS_PER_ROW.divisor)) * 32;
                #pragma unroll 1
                for (unsigned int i = 0; i < ROWS; i++) {
                    if (row < valid_rows) {
                        float values[32];
                        unsigned int words[16];
                        unsigned int tree[8];
                        if (k0 < (unsigned int)K) {
                            unsigned long long x_offset = x_row + (unsigned long long)k0;
                            {
                                uint32_t _iv_0_0;
                                uint32_t _iv_0_1;
                                uint32_t _iv_0_2;
                                uint32_t _iv_0_3;
                                uint32_t _iv_0_4;
                                uint32_t _iv_0_5;
                                uint32_t _iv_0_6;
                                uint32_t _iv_0_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(x + x_offset + (0))) : "memory");
                                words[0 + 0] = (uint32_t)_iv_0_0;
                                words[0 + 1] = (uint32_t)_iv_0_1;
                                words[0 + 2] = (uint32_t)_iv_0_2;
                                words[0 + 3] = (uint32_t)_iv_0_3;
                                words[0 + 4] = (uint32_t)_iv_0_4;
                                words[0 + 5] = (uint32_t)_iv_0_5;
                                words[0 + 6] = (uint32_t)_iv_0_6;
                                words[0 + 7] = (uint32_t)_iv_0_7;
                            }
                            {
                                uint32_t _iv_1_0;
                                uint32_t _iv_1_1;
                                uint32_t _iv_1_2;
                                uint32_t _iv_1_3;
                                uint32_t _iv_1_4;
                                uint32_t _iv_1_5;
                                uint32_t _iv_1_6;
                                uint32_t _iv_1_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(x + (x_offset + 16) + (0))) : "memory");
                                words[8 + 0] = (uint32_t)_iv_1_0;
                                words[8 + 1] = (uint32_t)_iv_1_1;
                                words[8 + 2] = (uint32_t)_iv_1_2;
                                words[8 + 3] = (uint32_t)_iv_1_3;
                                words[8 + 4] = (uint32_t)_iv_1_4;
                                words[8 + 5] = (uint32_t)_iv_1_5;
                                words[8 + 6] = (uint32_t)_iv_1_6;
                                words[8 + 7] = (uint32_t)_iv_1_7;
                            }
                        } else {
                            #pragma unroll
                            for (int word = 0; word < 16; word++) {
                                words[word] = 0;
                            }
                        }
#if CAKE_PROGRAM_VARIANT == 1
                        uint32_t _f16x2_abs_0;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_0) : "r"(words[0]));
                        uint32_t _f16x2_abs_16_0;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_0) : "r"(words[1]));
                        uint32_t _f16x2_max_0;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_0) : "r"(_f16x2_abs_0), "r"(_f16x2_abs_16_0));
                        tree[0] = _f16x2_max_0;
                        uint32_t _f16x2_abs_1;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_1) : "r"(words[2]));
                        uint32_t _f16x2_abs_16_1;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_1) : "r"(words[3]));
                        uint32_t _f16x2_max_1;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_1) : "r"(_f16x2_abs_1), "r"(_f16x2_abs_16_1));
                        tree[1] = _f16x2_max_1;
                        uint32_t _f16x2_abs_2;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_2) : "r"(words[4]));
                        uint32_t _f16x2_abs_16_2;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_2) : "r"(words[5]));
                        uint32_t _f16x2_max_2;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_2) : "r"(_f16x2_abs_2), "r"(_f16x2_abs_16_2));
                        tree[2] = _f16x2_max_2;
                        uint32_t _f16x2_abs_3;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_3) : "r"(words[6]));
                        uint32_t _f16x2_abs_16_3;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_3) : "r"(words[7]));
                        uint32_t _f16x2_max_3;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_3) : "r"(_f16x2_abs_3), "r"(_f16x2_abs_16_3));
                        tree[3] = _f16x2_max_3;
                        uint32_t _f16x2_abs_4;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_4) : "r"(words[8]));
                        uint32_t _f16x2_abs_16_4;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_4) : "r"(words[9]));
                        uint32_t _f16x2_max_4;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_4) : "r"(_f16x2_abs_4), "r"(_f16x2_abs_16_4));
                        tree[4] = _f16x2_max_4;
                        uint32_t _f16x2_abs_5;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_5) : "r"(words[10]));
                        uint32_t _f16x2_abs_16_5;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_5) : "r"(words[11]));
                        uint32_t _f16x2_max_5;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_5) : "r"(_f16x2_abs_5), "r"(_f16x2_abs_16_5));
                        tree[5] = _f16x2_max_5;
                        uint32_t _f16x2_abs_6;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_6) : "r"(words[12]));
                        uint32_t _f16x2_abs_16_6;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_6) : "r"(words[13]));
                        uint32_t _f16x2_max_6;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_6) : "r"(_f16x2_abs_6), "r"(_f16x2_abs_16_6));
                        tree[6] = _f16x2_max_6;
                        uint32_t _f16x2_abs_7;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_7) : "r"(words[14]));
                        uint32_t _f16x2_abs_16_7;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_7) : "r"(words[15]));
                        uint32_t _f16x2_max_7;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_7) : "r"(_f16x2_abs_7), "r"(_f16x2_abs_16_7));
                        tree[7] = _f16x2_max_7;
                        uint32_t _f16x2_max_8;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_8) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _f16x2_max_8;
                        uint32_t _f16x2_max_9;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_9) : "r"(tree[2]), "r"(tree[3]));
                        tree[1] = _f16x2_max_9;
                        uint32_t _f16x2_max_10;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_10) : "r"(tree[4]), "r"(tree[5]));
                        tree[2] = _f16x2_max_10;
                        uint32_t _f16x2_max_11;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_11) : "r"(tree[6]), "r"(tree[7]));
                        tree[3] = _f16x2_max_11;
                        uint32_t _f16x2_max_12;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_12) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _f16x2_max_12;
                        uint32_t _f16x2_max_13;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_13) : "r"(tree[2]), "r"(tree[3]));
                        tree[1] = _f16x2_max_13;
                        uint32_t _f16x2_max_14;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_14) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _f16x2_max_14;
                        uint32_t _f16x2_max_15;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_15) : "r"(tree[0]), "r"(tree[0] >> 16));
                        tree[0] = _f16x2_max_15;
                        float _cvt_f32_f16_2;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(tree[0] & 65535)));
                        unsigned int magnitude = __as_u32(_cvt_f32_f16_2) & 2147483647;
#else
                        uint32_t _bf16x2_abs_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(words[0]));
                        uint32_t _bf16x2_abs_16_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_0) : "r"(words[1]));
                        uint32_t _bf16x2_max_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_16_0));
                        tree[0] = _bf16x2_max_0;
                        uint32_t _bf16x2_abs_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(words[2]));
                        uint32_t _bf16x2_abs_16_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_1) : "r"(words[3]));
                        uint32_t _bf16x2_max_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_abs_1), "r"(_bf16x2_abs_16_1));
                        tree[1] = _bf16x2_max_1;
                        uint32_t _bf16x2_abs_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(words[4]));
                        uint32_t _bf16x2_abs_16_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_2) : "r"(words[5]));
                        uint32_t _bf16x2_max_2;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_abs_2), "r"(_bf16x2_abs_16_2));
                        tree[2] = _bf16x2_max_2;
                        uint32_t _bf16x2_abs_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(words[6]));
                        uint32_t _bf16x2_abs_16_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_3) : "r"(words[7]));
                        uint32_t _bf16x2_max_3;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_bf16x2_abs_3), "r"(_bf16x2_abs_16_3));
                        tree[3] = _bf16x2_max_3;
                        uint32_t _bf16x2_abs_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(words[8]));
                        uint32_t _bf16x2_abs_16_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_4) : "r"(words[9]));
                        uint32_t _bf16x2_max_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(_bf16x2_abs_4), "r"(_bf16x2_abs_16_4));
                        tree[4] = _bf16x2_max_4;
                        uint32_t _bf16x2_abs_5;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(words[10]));
                        uint32_t _bf16x2_abs_16_5;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_5) : "r"(words[11]));
                        uint32_t _bf16x2_max_5;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(_bf16x2_abs_5), "r"(_bf16x2_abs_16_5));
                        tree[5] = _bf16x2_max_5;
                        uint32_t _bf16x2_abs_6;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(words[12]));
                        uint32_t _bf16x2_abs_16_6;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_6) : "r"(words[13]));
                        uint32_t _bf16x2_max_6;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(_bf16x2_abs_6), "r"(_bf16x2_abs_16_6));
                        tree[6] = _bf16x2_max_6;
                        uint32_t _bf16x2_abs_7;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(words[14]));
                        uint32_t _bf16x2_abs_16_7;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_7) : "r"(words[15]));
                        uint32_t _bf16x2_max_7;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(_bf16x2_abs_7), "r"(_bf16x2_abs_16_7));
                        tree[7] = _bf16x2_max_7;
                        uint32_t _bf16x2_max_8;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _bf16x2_max_8;
                        uint32_t _bf16x2_max_9;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(tree[2]), "r"(tree[3]));
                        tree[1] = _bf16x2_max_9;
                        uint32_t _bf16x2_max_10;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(tree[4]), "r"(tree[5]));
                        tree[2] = _bf16x2_max_10;
                        uint32_t _bf16x2_max_11;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(tree[6]), "r"(tree[7]));
                        tree[3] = _bf16x2_max_11;
                        uint32_t _bf16x2_max_12;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _bf16x2_max_12;
                        uint32_t _bf16x2_max_13;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(tree[2]), "r"(tree[3]));
                        tree[1] = _bf16x2_max_13;
                        uint32_t _bf16x2_max_14;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(tree[0]), "r"(tree[1]));
                        tree[0] = _bf16x2_max_14;
                        uint32_t _bf16x2_max_15;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(tree[0]), "r"(tree[0] >> 16));
                        tree[0] = _bf16x2_max_15;
                        unsigned int magnitude = tree[0] << 16 & 2147483647;
#endif
                        int shifted = (int)(magnitude - 65011713) >> 23;
#if CAKE_PROGRAM_VARIANT == 1
                        int _max_3 = ((shifted) > (0) ? (shifted) : (0));
                        unsigned int finite_byte = (unsigned int)_max_3;
#else
                        int _max_2 = ((shifted) > (0) ? (shifted) : (0));
                        unsigned int finite_byte = (unsigned int)_max_2;
#endif
                        unsigned int saturated_byte = 254;
                        unsigned int scale_byte = ((magnitude > 2139095039) ? saturated_byte : finite_byte);
                        unsigned int inverse_nonzero_bits = 254 - scale_byte << 23;
                        unsigned int zero_bits = 0;
                        unsigned int inverse_bits = ((scale_byte == 0) ? zero_bits : inverse_nonzero_bits);
                        float inverse = 0.0f;
                        inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[_pair * 2]))
                                : "r"(words[_pair]));
#else
                            (&values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
                            (&values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[2 + _pair * 2]))
                                : "r"(words[1 + _pair]));
#else
                            (&values[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) << 16);
                            (&values[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[4 + _pair * 2]))
                                : "r"(words[2 + _pair]));
#else
                            (&values[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) << 16);
                            (&values[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[6 + _pair * 2]))
                                : "r"(words[3 + _pair]));
#else
                            (&values[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) << 16);
                            (&values[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[8 + _pair * 2]))
                                : "r"(words[4 + _pair]));
#else
                            (&values[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) << 16);
                            (&values[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[10 + _pair * 2]))
                                : "r"(words[5 + _pair]));
#else
                            (&values[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) << 16);
                            (&values[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[12 + _pair * 2]))
                                : "r"(words[6 + _pair]));
#else
                            (&values[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) << 16);
                            (&values[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[14 + _pair * 2]))
                                : "r"(words[7 + _pair]));
#else
                            (&values[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) << 16);
                            (&values[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[16 + _pair * 2]))
                                : "r"(words[8 + _pair]));
#else
                            (&values[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) << 16);
                            (&values[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[18 + _pair * 2]))
                                : "r"(words[9 + _pair]));
#else
                            (&values[18 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) << 16);
                            (&values[18 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[20 + _pair * 2]))
                                : "r"(words[10 + _pair]));
#else
                            (&values[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) << 16);
                            (&values[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[22 + _pair * 2]))
                                : "r"(words[11 + _pair]));
#else
                            (&values[22 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) << 16);
                            (&values[22 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[24 + _pair * 2]))
                                : "r"(words[12 + _pair]));
#else
                            (&values[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) << 16);
                            (&values[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[26 + _pair * 2]))
                                : "r"(words[13 + _pair]));
#else
                            (&values[26 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) << 16);
                            (&values[26 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[28 + _pair * 2]))
                                : "r"(words[14 + _pair]));
#else
                            (&values[28 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) << 16);
                            (&values[28 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) & 0xffff0000u);
#endif
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[30 + _pair * 2]))
                                : "r"(words[15 + _pair]));
#else
                            (&values[30 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) << 16);
                            (&values[30 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) & 0xffff0000u);
#endif
                        }
#if CAKE_PROGRAM_VARIANT == 1
                        const float2 _scale2_4 = {inverse, inverse};
#else
                        const float2 _scale2_3 = {inverse, inverse};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
#if CAKE_PROGRAM_VARIANT == 1
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_4);
#else
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_3);
#endif
                        unsigned long long q_offset = q_row + (unsigned long long)(_fast_div_r_0 * 32);
                        unsigned int _packed_0[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[0]), "f"(values[1]),
                                                   "f"(values[2]), "f"(values[3]));
                            _packed_0[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[4]), "f"(values[5]),
                                                   "f"(values[6]), "f"(values[7]));
                            _packed_0[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[8]), "f"(values[9]),
                                                   "f"(values[10]), "f"(values[11]));
                            _packed_0[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[12]), "f"(values[13]),
                                                   "f"(values[14]), "f"(values[15]));
                            _packed_0[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[16]), "f"(values[17]),
                                                   "f"(values[18]), "f"(values[19]));
                            _packed_0[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[20]), "f"(values[21]),
                                                   "f"(values[22]), "f"(values[23]));
                            _packed_0[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[24]), "f"(values[25]),
                                                   "f"(values[26]), "f"(values[27]));
                            _packed_0[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values[28]), "f"(values[29]),
                                                   "f"(values[30]), "f"(values[31]));
                            _packed_0[7] = _packed;
                        }
                        {
#if CAKE_PROGRAM_VARIANT == 1
                            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(_packed_0);
#else
                            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
#endif
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
#if CAKE_PROGRAM_VARIANT == 1
                                :: "l"((void*)(quantized + q_offset)), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
#else
                                :: "l"((void*)(quantized + q_offset)), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
#endif
                        }
                        unsigned int m_tile = row / 128;
                        unsigned int row_in_tile = row - m_tile * 128;
                        unsigned int k_tile = _fast_div_r_0 / 4;
                        unsigned int scale_col = _fast_div_r_0 - k_tile * 4;
                        unsigned long long tile_linear = ((unsigned long long)group * (unsigned long long)PM_TILES + (unsigned long long)m_tile) * (unsigned long long)PK_TILES + (unsigned long long)k_tile;
                        unsigned long long scale_offset = tile_linear * 512 + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4 + (unsigned long long)scale_col;
                        *(reinterpret_cast<unsigned char*>(scales + scale_offset) + (0)) = (unsigned char)(scale_byte);
                    }
                    row = row + SUBROWS;
                    x_row = x_row + (unsigned long long)SUBROWS * (unsigned long long)K;
                    q_row = q_row + (unsigned long long)SUBROWS * (unsigned long long)((uint32_t)(BLOCKS_PER_ROW.divisor)) * 32;
                }
            }
        }
    }
}

} // extern "C"
