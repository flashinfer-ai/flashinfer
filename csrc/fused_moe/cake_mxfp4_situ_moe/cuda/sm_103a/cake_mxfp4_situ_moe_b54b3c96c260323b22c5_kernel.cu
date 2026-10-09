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
#define THREADS 256

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

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_b54b3c96c260323b22c5(unsigned int* __restrict__ rows, int* __restrict__ perm, float* __restrict__ weights, unsigned int* __restrict__ out, int* __restrict__ narrow_count, int* __restrict__ wide_count, int chunks, int tasks, int narrow_tile, int wide_tile)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int task = blockIdx.x * 256 + tid;
    int wide_row0 = 2147483647;
    int has_wide = 1;
    int active = 0;
    if (task < tasks) {
        active = 1;
    }
    if (active != 0) {
        int token = task / chunks;
        int chunk = task - token * chunks;
        int row_words = chunks * 4;
        int chunk_words = chunk * 4;
        unsigned int src[32] = {0};
        float wgt[8] = {0};
        int permuted = perm[token * 8];
        if (permuted >= 0) {
            if (permuted < wide_row0) {
                int row = permuted;
                wgt[0] = weights[token * 8];
                int _vec_load_0[4];
                {
                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(rows + row * row_words + chunk_words);
                    int4 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    _vec_load_0[0 + 0] = _ivld_0.x;
                    _vec_load_0[0 + 1] = _ivld_0.y;
                    _vec_load_0[0 + 2] = _ivld_0.z;
                    _vec_load_0[0 + 3] = _ivld_0.w;
                }
                src[0] = __as_u32(_vec_load_0[0]);
                src[1] = __as_u32(_vec_load_0[1]);
                src[2] = __as_u32(_vec_load_0[2]);
                src[3] = __as_u32(_vec_load_0[3]);
            }
        }
        int permuted_0 = perm[token * 8 + 1];
        if (permuted_0 >= 0) {
            if (permuted_0 < wide_row0) {
                int row_1 = permuted_0;
                wgt[1] = weights[token * 8 + 1];
                int _vec_load_1[4];
                {
                    const int4* _ivptr_1 = reinterpret_cast<const int4*>(rows + row_1 * row_words + chunk_words);
                    int4 _ivld_1;
                    _ivld_1 = *_ivptr_1;
                    _vec_load_1[0 + 0] = _ivld_1.x;
                    _vec_load_1[0 + 1] = _ivld_1.y;
                    _vec_load_1[0 + 2] = _ivld_1.z;
                    _vec_load_1[0 + 3] = _ivld_1.w;
                }
                src[4] = __as_u32(_vec_load_1[0]);
                src[5] = __as_u32(_vec_load_1[1]);
                src[6] = __as_u32(_vec_load_1[2]);
                src[7] = __as_u32(_vec_load_1[3]);
            }
        }
        int permuted_1 = perm[token * 8 + 2];
        if (permuted_1 >= 0) {
            if (permuted_1 < wide_row0) {
                int row_2 = permuted_1;
                wgt[2] = weights[token * 8 + 2];
                int _vec_load_2[4];
                {
                    const int4* _ivptr_2 = reinterpret_cast<const int4*>(rows + row_2 * row_words + chunk_words);
                    int4 _ivld_2;
                    _ivld_2 = *_ivptr_2;
                    _vec_load_2[0 + 0] = _ivld_2.x;
                    _vec_load_2[0 + 1] = _ivld_2.y;
                    _vec_load_2[0 + 2] = _ivld_2.z;
                    _vec_load_2[0 + 3] = _ivld_2.w;
                }
                src[8] = __as_u32(_vec_load_2[0]);
                src[9] = __as_u32(_vec_load_2[1]);
                src[10] = __as_u32(_vec_load_2[2]);
                src[11] = __as_u32(_vec_load_2[3]);
            }
        }
        int permuted_2 = perm[token * 8 + 3];
        if (permuted_2 >= 0) {
            if (permuted_2 < wide_row0) {
                int row_3 = permuted_2;
                wgt[3] = weights[token * 8 + 3];
                int _vec_load_3[4];
                {
                    const int4* _ivptr_3 = reinterpret_cast<const int4*>(rows + row_3 * row_words + chunk_words);
                    int4 _ivld_3;
                    _ivld_3 = *_ivptr_3;
                    _vec_load_3[0 + 0] = _ivld_3.x;
                    _vec_load_3[0 + 1] = _ivld_3.y;
                    _vec_load_3[0 + 2] = _ivld_3.z;
                    _vec_load_3[0 + 3] = _ivld_3.w;
                }
                src[12] = __as_u32(_vec_load_3[0]);
                src[13] = __as_u32(_vec_load_3[1]);
                src[14] = __as_u32(_vec_load_3[2]);
                src[15] = __as_u32(_vec_load_3[3]);
            }
        }
        int permuted_3 = perm[token * 8 + 4];
        if (permuted_3 >= 0) {
            if (permuted_3 < wide_row0) {
                int row_4 = permuted_3;
                wgt[4] = weights[token * 8 + 4];
                int _vec_load_4[4];
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(rows + row_4 * row_words + chunk_words);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    _vec_load_4[0 + 0] = _ivld_4.x;
                    _vec_load_4[0 + 1] = _ivld_4.y;
                    _vec_load_4[0 + 2] = _ivld_4.z;
                    _vec_load_4[0 + 3] = _ivld_4.w;
                }
                src[16] = __as_u32(_vec_load_4[0]);
                src[17] = __as_u32(_vec_load_4[1]);
                src[18] = __as_u32(_vec_load_4[2]);
                src[19] = __as_u32(_vec_load_4[3]);
            }
        }
        int permuted_4 = perm[token * 8 + 5];
        if (permuted_4 >= 0) {
            if (permuted_4 < wide_row0) {
                int row_5 = permuted_4;
                wgt[5] = weights[token * 8 + 5];
                int _vec_load_5[4];
                {
                    const int4* _ivptr_5 = reinterpret_cast<const int4*>(rows + row_5 * row_words + chunk_words);
                    int4 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    _vec_load_5[0 + 0] = _ivld_5.x;
                    _vec_load_5[0 + 1] = _ivld_5.y;
                    _vec_load_5[0 + 2] = _ivld_5.z;
                    _vec_load_5[0 + 3] = _ivld_5.w;
                }
                src[20] = __as_u32(_vec_load_5[0]);
                src[21] = __as_u32(_vec_load_5[1]);
                src[22] = __as_u32(_vec_load_5[2]);
                src[23] = __as_u32(_vec_load_5[3]);
            }
        }
        int permuted_5 = perm[token * 8 + 6];
        if (permuted_5 >= 0) {
            if (permuted_5 < wide_row0) {
                int row_6 = permuted_5;
                wgt[6] = weights[token * 8 + 6];
                int _vec_load_6[4];
                {
                    const int4* _ivptr_6 = reinterpret_cast<const int4*>(rows + row_6 * row_words + chunk_words);
                    int4 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    _vec_load_6[0 + 0] = _ivld_6.x;
                    _vec_load_6[0 + 1] = _ivld_6.y;
                    _vec_load_6[0 + 2] = _ivld_6.z;
                    _vec_load_6[0 + 3] = _ivld_6.w;
                }
                src[24] = __as_u32(_vec_load_6[0]);
                src[25] = __as_u32(_vec_load_6[1]);
                src[26] = __as_u32(_vec_load_6[2]);
                src[27] = __as_u32(_vec_load_6[3]);
            }
        }
        int permuted_6 = perm[token * 8 + 7];
        if (permuted_6 >= 0) {
            if (permuted_6 < wide_row0) {
                int row_7 = permuted_6;
                wgt[7] = weights[token * 8 + 7];
                int _vec_load_7[4];
                {
                    const int4* _ivptr_7 = reinterpret_cast<const int4*>(rows + row_7 * row_words + chunk_words);
                    int4 _ivld_7;
                    _ivld_7 = *_ivptr_7;
                    _vec_load_7[0 + 0] = _ivld_7.x;
                    _vec_load_7[0 + 1] = _ivld_7.y;
                    _vec_load_7[0 + 2] = _ivld_7.z;
                    _vec_load_7[0 + 3] = _ivld_7.w;
                }
                src[28] = __as_u32(_vec_load_7[0]);
                src[29] = __as_u32(_vec_load_7[1]);
                src[30] = __as_u32(_vec_load_7[2]);
                src[31] = __as_u32(_vec_load_7[3]);
            }
        }
        float acc[8] = {0};
        int out_base = token * row_words + chunk_words;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned int slot_words[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                slot_words[j] = src[k * 4 + j];
            }
            float weight = wgt[k];
            unsigned int word = slot_words[0];
            float lo = __uint_as_float(word << 16);
            float hi = __uint_as_float(word & 4294901760u);
            float _fma_8 = __fmaf_rn(lo, weight, acc[0]);
            acc[0] = _fma_8;
            float _fma_9 = __fmaf_rn(hi, weight, acc[1]);
            acc[1] = _fma_9;
            unsigned int word_0 = slot_words[1];
            float lo_1 = __uint_as_float(word_0 << 16);
            float hi_2 = __uint_as_float(word_0 & 4294901760u);
            float _fma_10 = __fmaf_rn(lo_1, weight, acc[2]);
            acc[2] = _fma_10;
            float _fma_11 = __fmaf_rn(hi_2, weight, acc[3]);
            acc[3] = _fma_11;
            unsigned int word_3 = slot_words[2];
            float lo_4 = __uint_as_float(word_3 << 16);
            float hi_5 = __uint_as_float(word_3 & 4294901760u);
            float _fma_12 = __fmaf_rn(lo_4, weight, acc[4]);
            acc[4] = _fma_12;
            float _fma_13 = __fmaf_rn(hi_5, weight, acc[5]);
            acc[5] = _fma_13;
            unsigned int word_6 = slot_words[3];
            float lo_7 = __uint_as_float(word_6 << 16);
            float hi_8 = __uint_as_float(word_6 & 4294901760u);
            float _fma_14 = __fmaf_rn(lo_7, weight, acc[6]);
            acc[6] = _fma_14;
            float _fma_15 = __fmaf_rn(hi_8, weight, acc[7]);
            acc[7] = _fma_15;
        }
        unsigned int packed[4];
        #pragma unroll
        for (int j2 = 0; j2 < 4; j2++) {
            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(acc[2 * j2], acc[2 * j2 + 1]));
            packed[j2] = __as_u32(_bf16x2_0);
        }
        reinterpret_cast<int4*>(out + out_base)[0] = reinterpret_cast<int4*>(packed)[0];
    }
}

} // extern "C"
