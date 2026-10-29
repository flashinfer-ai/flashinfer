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

// This unit serves two programs: cake_grouped_mxfp8_quantize_377d98aae57150017fb0 (CAKE_PROGRAM_VARIANT 0, the default) and
// cake_grouped_mxfp8_quantize_93255ff718a35b75ab2b (CAKE_PROGRAM_VARIANT 1).  CAKE_PROGRAM_KERNEL names the kernel it defines.  A build
// selects the second program with the compile definitions
// -DCAKE_PROGRAM_VARIANT=1 -DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_93255ff718a35b75ab2b.
#ifndef CAKE_PROGRAM_VARIANT
#define CAKE_PROGRAM_VARIANT 0
#endif
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_grouped_mxfp8_quantize_377d98aae57150017fb0
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
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 5

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
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
        unsigned int own_rows = valid_rows;
        unsigned int partner = group;
        unsigned int partner_rows = 0;
        if (ROWS > 1 && valid_rows < (unsigned int)M) {
            if (tid == 0) {
                unsigned int far_group = group + 2;
                if (far_group < (unsigned int)gridDim.z) {
                    unsigned int sparse_rows = (unsigned int)((unsigned long long)M * 3 / 8);
                    int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const int*>(mask + far_group);
                    }
                    unsigned int far_valid = (((unsigned int)_vec_load_3[0]) < ((unsigned int)M) ? ((unsigned int)_vec_load_3[0]) : ((unsigned int)M));
                    unsigned int far_first = tile * (ROWS * SUBROWS);
                    if (far_first < far_valid && far_valid <= sparse_rows) {
                        unsigned int far_k_lo = (unsigned int)blockIdx.x * 8192;
                        unsigned int far_rows = ((far_valid - far_first) < (ROWS * SUBROWS) ? (far_valid - far_first) : (ROWS * SUBROWS));
                        unsigned int _min_0 = (((unsigned int)K - far_k_lo) < (8192) ? ((unsigned int)K - far_k_lo) : (8192));
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (((unsigned long long)far_group * (unsigned long long)M + (unsigned long long)far_first) * (unsigned long long)K + (unsigned long long)far_k_lo)), "r"((uint32_t)(((far_rows - 1) * (unsigned int)K + _min_0) * 2)) : "memory");
                    }
                }
            }
            if (valid_rows <= 32) {
                unsigned int groups = gridDim.z;
                unsigned int pairs = ((12) < (groups >> 1) ? (12) : (groups >> 1));
                unsigned int cut = groups - pairs;
                if (group < pairs || group >= cut) {
                    partner = ((group < pairs) ? group + cut : group - cut);
                    int _vec_load_1[1];
                    {
                        _vec_load_1[0] = *reinterpret_cast<const int*>(mask + partner);
                    }
                    unsigned int partner_word = (unsigned int)_vec_load_1[0];
                    bool donate = partner_word < (unsigned int)M && partner_word <= 32 && tile * (ROWS * SUBROWS) < partner_word;
                    if (donate) {
                        if (group < pairs) {
                            if (tid == 0) {
                                unsigned int p_first = tile * (ROWS * SUBROWS);
                                unsigned int p_k_lo = (unsigned int)blockIdx.x * 8192;
                                unsigned int p_rows = ((partner_word - p_first) < (ROWS * SUBROWS) ? (partner_word - p_first) : (ROWS * SUBROWS));
                                unsigned int _min_1 = (((unsigned int)K - p_k_lo) < (8192) ? ((unsigned int)K - p_k_lo) : (8192));
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (((unsigned long long)partner * (unsigned long long)M + (unsigned long long)p_first) * (unsigned long long)K + (unsigned long long)p_k_lo)), "r"((uint32_t)(((p_rows - 1) * (unsigned int)K + _min_1) * 2)) : "memory");
                            }
                            partner_rows = partner_word;
                        } else {
                            own_rows = 0;
                        }
                    }
                }
            }
        }
        unsigned int tid_0 = (unsigned int)tid;
        unsigned int flat = (unsigned int)blockIdx.x * 256 + tid_0;
        uint32_t _fast_div_q_0 = (BLOCKS_PER_ROW.divisor != 1) ? (__umulhi((uint32_t)(flat), BLOCKS_PER_ROW.multiplier) >> BLOCKS_PER_ROW.shift_right) : (uint32_t)(flat);
        uint32_t _fast_div_r_0 = (uint32_t)(flat) - _fast_div_q_0 * (uint32_t)(BLOCKS_PER_ROW.divisor);
        if (_fast_div_q_0 < SUBROWS) {
            unsigned int first_row = tile * (ROWS * SUBROWS) + _fast_div_q_0;
            if (first_row < own_rows) {
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
                                uint32_t _iv_2_0;
                                uint32_t _iv_2_1;
                                uint32_t _iv_2_2;
                                uint32_t _iv_2_3;
                                uint32_t _iv_2_4;
                                uint32_t _iv_2_5;
                                uint32_t _iv_2_6;
                                uint32_t _iv_2_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(x + x_offset + (0))) : "memory");
                                words[0 + 0] = (uint32_t)_iv_2_0;
                                words[0 + 1] = (uint32_t)_iv_2_1;
                                words[0 + 2] = (uint32_t)_iv_2_2;
                                words[0 + 3] = (uint32_t)_iv_2_3;
                                words[0 + 4] = (uint32_t)_iv_2_4;
                                words[0 + 5] = (uint32_t)_iv_2_5;
                                words[0 + 6] = (uint32_t)_iv_2_6;
                                words[0 + 7] = (uint32_t)_iv_2_7;
                            }
                            {
                                uint32_t _iv_3_0;
                                uint32_t _iv_3_1;
                                uint32_t _iv_3_2;
                                uint32_t _iv_3_3;
                                uint32_t _iv_3_4;
                                uint32_t _iv_3_5;
                                uint32_t _iv_3_6;
                                uint32_t _iv_3_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(x + (x_offset + 16) + (0))) : "memory");
                                words[8 + 0] = (uint32_t)_iv_3_0;
                                words[8 + 1] = (uint32_t)_iv_3_1;
                                words[8 + 2] = (uint32_t)_iv_3_2;
                                words[8 + 3] = (uint32_t)_iv_3_3;
                                words[8 + 4] = (uint32_t)_iv_3_4;
                                words[8 + 5] = (uint32_t)_iv_3_5;
                                words[8 + 6] = (uint32_t)_iv_3_6;
                                words[8 + 7] = (uint32_t)_iv_3_7;
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
                        float _cvt_f32_f16_4;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(tree[0] & 65535)));
                        unsigned int magnitude = __as_u32(_cvt_f32_f16_4) & 2147483647;
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
                        int _max_5 = ((shifted) > (0) ? (shifted) : (0));
                        unsigned int finite_byte = (unsigned int)_max_5;
#else
                        int _max_4 = ((shifted) > (0) ? (shifted) : (0));
                        unsigned int finite_byte = (unsigned int)_max_4;
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
                        const float2 _scale2_6 = {inverse, inverse};
#else
                        const float2 _scale2_5 = {inverse, inverse};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
#if CAKE_PROGRAM_VARIANT == 1
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_6);
#else
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_5);
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
                            const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(_packed_0);
#else
                            const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(_packed_0);
#endif
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
#if CAKE_PROGRAM_VARIANT == 1
                                :: "l"((void*)(quantized + q_offset)), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
#else
                                :: "l"((void*)(quantized + q_offset)), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
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
            if (first_row < partner_rows) {
                unsigned int k0_p = _fast_div_r_0 * 32;
                unsigned int row_p = first_row;
                unsigned long long global_row_p = (unsigned long long)partner * (unsigned long long)M + (unsigned long long)row_p;
                unsigned long long x_row_p = global_row_p * (unsigned long long)K;
                unsigned long long q_row_p = global_row_p * (unsigned long long)((uint32_t)(BLOCKS_PER_ROW.divisor)) * 32;
                #pragma unroll 1
                for (unsigned int i_p = 0; i_p < ROWS; i_p++) {
                    if (row_p < partner_rows) {
                        float values_1[32];
                        unsigned int words_1[16];
                        unsigned int tree_1[8];
                        if (k0_p < (unsigned int)K) {
                            unsigned long long x_offset_1 = x_row_p + (unsigned long long)k0_p;
                            {
#if CAKE_PROGRAM_VARIANT == 1
#else
                                uint32_t _iv_7_0;
                                uint32_t _iv_7_1;
                                uint32_t _iv_7_2;
                                uint32_t _iv_7_3;
                                uint32_t _iv_7_4;
                                uint32_t _iv_7_5;
                                uint32_t _iv_7_6;
                                uint32_t _iv_7_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(x + x_offset_1 + (0))) : "memory");
                                words_1[0 + 0] = (uint32_t)_iv_7_0;
                                words_1[0 + 1] = (uint32_t)_iv_7_1;
                                words_1[0 + 2] = (uint32_t)_iv_7_2;
                                words_1[0 + 3] = (uint32_t)_iv_7_3;
                                words_1[0 + 4] = (uint32_t)_iv_7_4;
                                words_1[0 + 5] = (uint32_t)_iv_7_5;
                                words_1[0 + 6] = (uint32_t)_iv_7_6;
                                words_1[0 + 7] = (uint32_t)_iv_7_7;
                            }
                            {
#endif
                                uint32_t _iv_8_0;
                                uint32_t _iv_8_1;
                                uint32_t _iv_8_2;
                                uint32_t _iv_8_3;
                                uint32_t _iv_8_4;
                                uint32_t _iv_8_5;
                                uint32_t _iv_8_6;
                                uint32_t _iv_8_7;
#if CAKE_PROGRAM_VARIANT == 1
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_8_0), "=r"(_iv_8_1), "=r"(_iv_8_2), "=r"(_iv_8_3), "=r"(_iv_8_4), "=r"(_iv_8_5), "=r"(_iv_8_6), "=r"(_iv_8_7) : "l"((const void*)(x + x_offset_1 + (0))) : "memory");
                                words_1[0 + 0] = (uint32_t)_iv_8_0;
                                words_1[0 + 1] = (uint32_t)_iv_8_1;
                                words_1[0 + 2] = (uint32_t)_iv_8_2;
                                words_1[0 + 3] = (uint32_t)_iv_8_3;
                                words_1[0 + 4] = (uint32_t)_iv_8_4;
                                words_1[0 + 5] = (uint32_t)_iv_8_5;
                                words_1[0 + 6] = (uint32_t)_iv_8_6;
                                words_1[0 + 7] = (uint32_t)_iv_8_7;
                            }
                            {
                                uint32_t _iv_9_0;
                                uint32_t _iv_9_1;
                                uint32_t _iv_9_2;
                                uint32_t _iv_9_3;
                                uint32_t _iv_9_4;
                                uint32_t _iv_9_5;
                                uint32_t _iv_9_6;
                                uint32_t _iv_9_7;
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_9_0), "=r"(_iv_9_1), "=r"(_iv_9_2), "=r"(_iv_9_3), "=r"(_iv_9_4), "=r"(_iv_9_5), "=r"(_iv_9_6), "=r"(_iv_9_7) : "l"((const void*)(x + (x_offset_1 + 16) + (0))) : "memory");
                                words_1[8 + 0] = (uint32_t)_iv_9_0;
                                words_1[8 + 1] = (uint32_t)_iv_9_1;
                                words_1[8 + 2] = (uint32_t)_iv_9_2;
                                words_1[8 + 3] = (uint32_t)_iv_9_3;
                                words_1[8 + 4] = (uint32_t)_iv_9_4;
                                words_1[8 + 5] = (uint32_t)_iv_9_5;
                                words_1[8 + 6] = (uint32_t)_iv_9_6;
                                words_1[8 + 7] = (uint32_t)_iv_9_7;
#else
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_8_0), "=r"(_iv_8_1), "=r"(_iv_8_2), "=r"(_iv_8_3), "=r"(_iv_8_4), "=r"(_iv_8_5), "=r"(_iv_8_6), "=r"(_iv_8_7) : "l"((const void*)(x + (x_offset_1 + 16) + (0))) : "memory");
                                words_1[8 + 0] = (uint32_t)_iv_8_0;
                                words_1[8 + 1] = (uint32_t)_iv_8_1;
                                words_1[8 + 2] = (uint32_t)_iv_8_2;
                                words_1[8 + 3] = (uint32_t)_iv_8_3;
                                words_1[8 + 4] = (uint32_t)_iv_8_4;
                                words_1[8 + 5] = (uint32_t)_iv_8_5;
                                words_1[8 + 6] = (uint32_t)_iv_8_6;
                                words_1[8 + 7] = (uint32_t)_iv_8_7;
#endif
                            }
                        } else {
                            #pragma unroll
                            for (int word_1 = 0; word_1 < 16; word_1++) {
                                words_1[word_1] = 0;
                            }
                        }
#if CAKE_PROGRAM_VARIANT == 1
                        uint32_t _f16x2_abs_8;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_8) : "r"(words_1[0]));
                        uint32_t _f16x2_abs_16_8;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_8) : "r"(words_1[1]));
                        uint32_t _f16x2_max_14_1;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_14_1) : "r"(_f16x2_abs_8), "r"(_f16x2_abs_16_8));
                        tree_1[0] = _f16x2_max_14_1;
                        uint32_t _f16x2_abs_9;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_9) : "r"(words_1[2]));
                        uint32_t _f16x2_abs_16_9;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_9) : "r"(words_1[3]));
                        uint32_t _f16x2_max_15_1;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_15_1) : "r"(_f16x2_abs_9), "r"(_f16x2_abs_16_9));
                        tree_1[1] = _f16x2_max_15_1;
                        uint32_t _f16x2_abs_10;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_10) : "r"(words_1[4]));
                        uint32_t _f16x2_abs_16_10;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_10) : "r"(words_1[5]));
                        uint32_t _f16x2_max_16;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_16) : "r"(_f16x2_abs_10), "r"(_f16x2_abs_16_10));
                        tree_1[2] = _f16x2_max_16;
                        uint32_t _f16x2_abs_11;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_11) : "r"(words_1[6]));
                        uint32_t _f16x2_abs_16_11;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_11) : "r"(words_1[7]));
                        uint32_t _f16x2_max_17;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_17) : "r"(_f16x2_abs_11), "r"(_f16x2_abs_16_11));
                        tree_1[3] = _f16x2_max_17;
                        uint32_t _f16x2_abs_12;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_12) : "r"(words_1[8]));
                        uint32_t _f16x2_abs_16_12;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_12) : "r"(words_1[9]));
                        uint32_t _f16x2_max_18;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_18) : "r"(_f16x2_abs_12), "r"(_f16x2_abs_16_12));
                        tree_1[4] = _f16x2_max_18;
                        uint32_t _f16x2_abs_13;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_13) : "r"(words_1[10]));
                        uint32_t _f16x2_abs_16_13;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_13) : "r"(words_1[11]));
                        uint32_t _f16x2_max_19;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_19) : "r"(_f16x2_abs_13), "r"(_f16x2_abs_16_13));
                        tree_1[5] = _f16x2_max_19;
                        uint32_t _f16x2_abs_14;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_14) : "r"(words_1[12]));
                        uint32_t _f16x2_abs_16_14;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_14) : "r"(words_1[13]));
                        uint32_t _f16x2_max_20;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_20) : "r"(_f16x2_abs_14), "r"(_f16x2_abs_16_14));
                        tree_1[6] = _f16x2_max_20;
                        uint32_t _f16x2_abs_15;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_15) : "r"(words_1[14]));
                        uint32_t _f16x2_abs_16_15;
                        asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_16_15) : "r"(words_1[15]));
                        uint32_t _f16x2_max_21;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_21) : "r"(_f16x2_abs_15), "r"(_f16x2_abs_16_15));
                        tree_1[7] = _f16x2_max_21;
                        uint32_t _f16x2_max_22;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_22) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _f16x2_max_22;
                        uint32_t _f16x2_max_23;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_23) : "r"(tree_1[2]), "r"(tree_1[3]));
                        tree_1[1] = _f16x2_max_23;
                        uint32_t _f16x2_max_24;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_24) : "r"(tree_1[4]), "r"(tree_1[5]));
                        tree_1[2] = _f16x2_max_24;
                        uint32_t _f16x2_max_25;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_25) : "r"(tree_1[6]), "r"(tree_1[7]));
                        tree_1[3] = _f16x2_max_25;
                        uint32_t _f16x2_max_26;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_26) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _f16x2_max_26;
                        uint32_t _f16x2_max_27;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_27) : "r"(tree_1[2]), "r"(tree_1[3]));
                        tree_1[1] = _f16x2_max_27;
                        uint32_t _f16x2_max_14_0;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_14_0) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _f16x2_max_14_0;
                        uint32_t _f16x2_max_15_0;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_15_0) : "r"(tree_1[0]), "r"(tree_1[0] >> 16));
                        tree_1[0] = _f16x2_max_15_0;
                        float _cvt_f32_f16_10;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(tree_1[0] & 65535)));
                        unsigned int magnitude_1 = __as_u32(_cvt_f32_f16_10) & 2147483647;
#else
                        uint32_t _bf16x2_abs_8;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(words_1[0]));
                        uint32_t _bf16x2_abs_16_8;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_8) : "r"(words_1[1]));
                        uint32_t _bf16x2_max_14_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14_1) : "r"(_bf16x2_abs_8), "r"(_bf16x2_abs_16_8));
                        tree_1[0] = _bf16x2_max_14_1;
                        uint32_t _bf16x2_abs_9;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(words_1[2]));
                        uint32_t _bf16x2_abs_16_9;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_9) : "r"(words_1[3]));
                        uint32_t _bf16x2_max_15_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15_1) : "r"(_bf16x2_abs_9), "r"(_bf16x2_abs_16_9));
                        tree_1[1] = _bf16x2_max_15_1;
                        uint32_t _bf16x2_abs_10;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(words_1[4]));
                        uint32_t _bf16x2_abs_16_10;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_10) : "r"(words_1[5]));
                        uint32_t _bf16x2_max_16;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(_bf16x2_abs_10), "r"(_bf16x2_abs_16_10));
                        tree_1[2] = _bf16x2_max_16;
                        uint32_t _bf16x2_abs_11;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(words_1[6]));
                        uint32_t _bf16x2_abs_16_11;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_11) : "r"(words_1[7]));
                        uint32_t _bf16x2_max_17;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(_bf16x2_abs_11), "r"(_bf16x2_abs_16_11));
                        tree_1[3] = _bf16x2_max_17;
                        uint32_t _bf16x2_abs_12;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(words_1[8]));
                        uint32_t _bf16x2_abs_16_12;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_12) : "r"(words_1[9]));
                        uint32_t _bf16x2_max_18;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(_bf16x2_abs_12), "r"(_bf16x2_abs_16_12));
                        tree_1[4] = _bf16x2_max_18;
                        uint32_t _bf16x2_abs_13;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(words_1[10]));
                        uint32_t _bf16x2_abs_16_13;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_13) : "r"(words_1[11]));
                        uint32_t _bf16x2_max_19;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(_bf16x2_abs_13), "r"(_bf16x2_abs_16_13));
                        tree_1[5] = _bf16x2_max_19;
                        uint32_t _bf16x2_abs_14;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(words_1[12]));
                        uint32_t _bf16x2_abs_16_14;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_14) : "r"(words_1[13]));
                        uint32_t _bf16x2_max_20;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(_bf16x2_abs_14), "r"(_bf16x2_abs_16_14));
                        tree_1[6] = _bf16x2_max_20;
                        uint32_t _bf16x2_abs_15;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(words_1[14]));
                        uint32_t _bf16x2_abs_16_15;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16_15) : "r"(words_1[15]));
                        uint32_t _bf16x2_max_21;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(_bf16x2_abs_15), "r"(_bf16x2_abs_16_15));
                        tree_1[7] = _bf16x2_max_21;
                        uint32_t _bf16x2_max_22;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _bf16x2_max_22;
                        uint32_t _bf16x2_max_23;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(tree_1[2]), "r"(tree_1[3]));
                        tree_1[1] = _bf16x2_max_23;
                        uint32_t _bf16x2_max_24;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(tree_1[4]), "r"(tree_1[5]));
                        tree_1[2] = _bf16x2_max_24;
                        uint32_t _bf16x2_max_25;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(tree_1[6]), "r"(tree_1[7]));
                        tree_1[3] = _bf16x2_max_25;
                        uint32_t _bf16x2_max_26;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _bf16x2_max_26;
                        uint32_t _bf16x2_max_27;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(tree_1[2]), "r"(tree_1[3]));
                        tree_1[1] = _bf16x2_max_27;
                        uint32_t _bf16x2_max_14_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14_0) : "r"(tree_1[0]), "r"(tree_1[1]));
                        tree_1[0] = _bf16x2_max_14_0;
                        uint32_t _bf16x2_max_15_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15_0) : "r"(tree_1[0]), "r"(tree_1[0] >> 16));
                        tree_1[0] = _bf16x2_max_15_0;
                        unsigned int magnitude_1 = tree_1[0] << 16 & 2147483647;
#endif
                        int shifted_1 = (int)(magnitude_1 - 65011713) >> 23;
#if CAKE_PROGRAM_VARIANT == 1
                        int _max_11 = ((shifted_1) > (0) ? (shifted_1) : (0));
                        unsigned int finite_byte_1 = (unsigned int)_max_11;
#else
                        int _max_9 = ((shifted_1) > (0) ? (shifted_1) : (0));
                        unsigned int finite_byte_1 = (unsigned int)_max_9;
#endif
                        unsigned int saturated_byte_1 = 254;
                        unsigned int scale_byte_1 = ((magnitude_1 > 2139095039) ? saturated_byte_1 : finite_byte_1);
                        unsigned int inverse_nonzero_bits_1 = 254 - scale_byte_1 << 23;
                        unsigned int zero_bits_1 = 0;
                        unsigned int inverse_bits_1 = ((scale_byte_1 == 0) ? zero_bits_1 : inverse_nonzero_bits_1);
                        float inverse_1 = 0.0f;
                        inverse_1 = reinterpret_cast<float*>(&inverse_bits_1)[0];
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[_pair * 2]))
                                : "r"(words_1[_pair]));
#else
                            (&values_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) << 16);
                            (&values_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[2 + _pair * 2]))
                                : "r"(words_1[1 + _pair]));
#else
                            (&values_1[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[1 + _pair]) << 16);
                            (&values_1[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[1 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[4 + _pair * 2]))
                                : "r"(words_1[2 + _pair]));
#else
                            (&values_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[2 + _pair]) << 16);
                            (&values_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[2 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[6 + _pair * 2]))
                                : "r"(words_1[3 + _pair]));
#else
                            (&values_1[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[3 + _pair]) << 16);
                            (&values_1[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[3 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[8 + _pair * 2]))
                                : "r"(words_1[4 + _pair]));
#else
                            (&values_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[4 + _pair]) << 16);
                            (&values_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[4 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[10 + _pair * 2]))
                                : "r"(words_1[5 + _pair]));
#else
                            (&values_1[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[5 + _pair]) << 16);
                            (&values_1[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[5 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[12 + _pair * 2]))
                                : "r"(words_1[6 + _pair]));
#else
                            (&values_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[6 + _pair]) << 16);
                            (&values_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[6 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[14 + _pair * 2]))
                                : "r"(words_1[7 + _pair]));
#else
                            (&values_1[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[7 + _pair]) << 16);
                            (&values_1[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[7 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[16 + _pair * 2]))
                                : "r"(words_1[8 + _pair]));
#else
                            (&values_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[8 + _pair]) << 16);
                            (&values_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[8 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[18 + _pair * 2]))
                                : "r"(words_1[9 + _pair]));
#else
                            (&values_1[18 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[9 + _pair]) << 16);
                            (&values_1[18 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[9 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[20 + _pair * 2]))
                                : "r"(words_1[10 + _pair]));
#else
                            (&values_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[10 + _pair]) << 16);
                            (&values_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[10 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[22 + _pair * 2]))
                                : "r"(words_1[11 + _pair]));
#else
                            (&values_1[22 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[11 + _pair]) << 16);
                            (&values_1[22 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[11 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[24 + _pair * 2]))
                                : "r"(words_1[12 + _pair]));
#else
                            (&values_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[12 + _pair]) << 16);
                            (&values_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[12 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[26 + _pair * 2]))
                                : "r"(words_1[13 + _pair]));
#else
                            (&values_1[26 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[13 + _pair]) << 16);
                            (&values_1[26 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[13 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[28 + _pair * 2]))
                                : "r"(words_1[14 + _pair]));
#else
                            (&values_1[28 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[14 + _pair]) << 16);
                            (&values_1[28 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[14 + _pair]) & 0xffff0000u);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values_1[30 + _pair * 2]))
                                : "r"(words_1[15 + _pair]));
#else
                            (&values_1[30 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[15 + _pair]) << 16);
                            (&values_1[30 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[15 + _pair]) & 0xffff0000u);
#endif
                        }
#if CAKE_PROGRAM_VARIANT == 1
                        const float2 _scale2_12 = {inverse_1, inverse_1};
#else
                        const float2 _scale2_10 = {inverse_1, inverse_1};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
#if CAKE_PROGRAM_VARIANT == 1
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values_1)[_ls], _scale2_12);
#else
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values_1)[_ls], _scale2_10);
#endif
                        unsigned long long q_offset_1 = q_row_p + (unsigned long long)(_fast_div_r_0 * 32);
                        unsigned int _packed_0_1[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(values_1[0]), "f"(values_1[1]),
                                                   "f"(values_1[2]), "f"(values_1[3]));
                            _packed_0_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[4]), "f"(values_1[5]),
                                                   "f"(values_1[6]), "f"(values_1[7]));
                            _packed_0_1[1] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[8]), "f"(values_1[9]),
                                                   "f"(values_1[10]), "f"(values_1[11]));
                            _packed_0_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[12]), "f"(values_1[13]),
                                                   "f"(values_1[14]), "f"(values_1[15]));
                            _packed_0_1[3] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[16]), "f"(values_1[17]),
                                                   "f"(values_1[18]), "f"(values_1[19]));
                            _packed_0_1[4] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[20]), "f"(values_1[21]),
                                                   "f"(values_1[22]), "f"(values_1[23]));
                            _packed_0_1[5] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[24]), "f"(values_1[25]),
                                                   "f"(values_1[26]), "f"(values_1[27]));
                            _packed_0_1[6] = _packed;
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
                                : "=r"(_packed) : "f"(values_1[28]), "f"(values_1[29]),
                                                   "f"(values_1[30]), "f"(values_1[31]));
                            _packed_0_1[7] = _packed;
                        }
                        {
#if CAKE_PROGRAM_VARIANT == 1
                            const unsigned* _raw_stv8_13 = reinterpret_cast<const unsigned*>(_packed_0_1);
#else
                            const unsigned* _raw_stv8_11 = reinterpret_cast<const unsigned*>(_packed_0_1);
#endif
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
#if CAKE_PROGRAM_VARIANT == 1
                                :: "l"((void*)(quantized + q_offset_1)), "r"(_raw_stv8_13[0]), "r"(_raw_stv8_13[1]), "r"(_raw_stv8_13[2]), "r"(_raw_stv8_13[3]), "r"(_raw_stv8_13[4]), "r"(_raw_stv8_13[5]), "r"(_raw_stv8_13[6]), "r"(_raw_stv8_13[7]) : "memory");
#else
                                :: "l"((void*)(quantized + q_offset_1)), "r"(_raw_stv8_11[0]), "r"(_raw_stv8_11[1]), "r"(_raw_stv8_11[2]), "r"(_raw_stv8_11[3]), "r"(_raw_stv8_11[4]), "r"(_raw_stv8_11[5]), "r"(_raw_stv8_11[6]), "r"(_raw_stv8_11[7]) : "memory");
#endif
                        }
                        unsigned int m_tile_1 = row_p / 128;
                        unsigned int row_in_tile_1 = row_p - m_tile_1 * 128;
                        unsigned int k_tile_1 = _fast_div_r_0 / 4;
                        unsigned int scale_col_1 = _fast_div_r_0 - k_tile_1 * 4;
                        unsigned long long tile_linear_1 = ((unsigned long long)partner * (unsigned long long)PM_TILES + (unsigned long long)m_tile_1) * (unsigned long long)PK_TILES + (unsigned long long)k_tile_1;
                        unsigned long long scale_offset_1 = tile_linear_1 * 512 + (unsigned long long)(row_in_tile_1 & 31) * 16 + (unsigned long long)(row_in_tile_1 >> 5) * 4 + (unsigned long long)scale_col_1;
                        *(reinterpret_cast<unsigned char*>(scales + scale_offset_1) + (0)) = (unsigned char)(scale_byte_1);
                    }
                    row_p = row_p + SUBROWS;
                    x_row_p = x_row_p + (unsigned long long)SUBROWS * (unsigned long long)K;
                    q_row_p = q_row_p + (unsigned long long)SUBROWS * (unsigned long long)((uint32_t)(BLOCKS_PER_ROW.divisor)) * 32;
                }
            }
        }
    }
}

} // extern "C"
