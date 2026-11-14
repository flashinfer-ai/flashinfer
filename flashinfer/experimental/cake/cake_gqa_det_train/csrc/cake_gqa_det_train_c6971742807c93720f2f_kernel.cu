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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gqa_det_train_c6971742807c93720f2f(float* __restrict__ dk_win, float* __restrict__ dv_win, int* __restrict__ perm, int* __restrict__ dst_rows, int* __restrict__ run_end, float* __restrict__ dk_out, float* __restrict__ dv_out, int T_k, int h_kv)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int item = blockIdx.x * 8 + warp;
    int i = item / h_kv;
    int h = item % h_kv;
    long long col = (long long)(lane * 8);
    if (i < T_k) {
        int r = dst_rows[i];
        int first = 1;
        if (i > 0) {
            if (dst_rows[i - 1] == r) {
                first = 0;
            }
        }
        if (first != 0) {
            int end = run_end[i];
            long long out = ((long long)r * (long long)h_kv + (long long)h) * 256ll + col;
            float ak[8];
            float av[8];
            {
                unsigned _ldv8_0_0;
                unsigned _ldv8_0_1;
                unsigned _ldv8_0_2;
                unsigned _ldv8_0_3;
                unsigned _ldv8_0_4;
                unsigned _ldv8_0_5;
                unsigned _ldv8_0_6;
                unsigned _ldv8_0_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(dk_out + out + (0))) : "memory");
                ak[0 + 0] = __uint_as_float(_ldv8_0_0);
                ak[0 + 1] = __uint_as_float(_ldv8_0_1);
                ak[0 + 2] = __uint_as_float(_ldv8_0_2);
                ak[0 + 3] = __uint_as_float(_ldv8_0_3);
                ak[0 + 4] = __uint_as_float(_ldv8_0_4);
                ak[0 + 5] = __uint_as_float(_ldv8_0_5);
                ak[0 + 6] = __uint_as_float(_ldv8_0_6);
                ak[0 + 7] = __uint_as_float(_ldv8_0_7);
            }
            {
                unsigned _ldv8_1_0;
                unsigned _ldv8_1_1;
                unsigned _ldv8_1_2;
                unsigned _ldv8_1_3;
                unsigned _ldv8_1_4;
                unsigned _ldv8_1_5;
                unsigned _ldv8_1_6;
                unsigned _ldv8_1_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(dv_out + out + (0))) : "memory");
                av[0 + 0] = __uint_as_float(_ldv8_1_0);
                av[0 + 1] = __uint_as_float(_ldv8_1_1);
                av[0 + 2] = __uint_as_float(_ldv8_1_2);
                av[0 + 3] = __uint_as_float(_ldv8_1_3);
                av[0 + 4] = __uint_as_float(_ldv8_1_4);
                av[0 + 5] = __uint_as_float(_ldv8_1_5);
                av[0 + 6] = __uint_as_float(_ldv8_1_6);
                av[0 + 7] = __uint_as_float(_ldv8_1_7);
            }
            #pragma unroll 1
            for (int j = i; j < end; j++) {
                long long src = ((long long)perm[j] * (long long)h_kv + (long long)h) * 256ll + col;
                float wk[8];
                float wv[8];
                {
                    unsigned _ldv8_2_0;
                    unsigned _ldv8_2_1;
                    unsigned _ldv8_2_2;
                    unsigned _ldv8_2_3;
                    unsigned _ldv8_2_4;
                    unsigned _ldv8_2_5;
                    unsigned _ldv8_2_6;
                    unsigned _ldv8_2_7;
                    asm volatile(
                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(dk_win + src + (0))) : "memory");
                    wk[0 + 0] = __uint_as_float(_ldv8_2_0);
                    wk[0 + 1] = __uint_as_float(_ldv8_2_1);
                    wk[0 + 2] = __uint_as_float(_ldv8_2_2);
                    wk[0 + 3] = __uint_as_float(_ldv8_2_3);
                    wk[0 + 4] = __uint_as_float(_ldv8_2_4);
                    wk[0 + 5] = __uint_as_float(_ldv8_2_5);
                    wk[0 + 6] = __uint_as_float(_ldv8_2_6);
                    wk[0 + 7] = __uint_as_float(_ldv8_2_7);
                }
                {
                    unsigned _ldv8_3_0;
                    unsigned _ldv8_3_1;
                    unsigned _ldv8_3_2;
                    unsigned _ldv8_3_3;
                    unsigned _ldv8_3_4;
                    unsigned _ldv8_3_5;
                    unsigned _ldv8_3_6;
                    unsigned _ldv8_3_7;
                    asm volatile(
                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(dv_win + src + (0))) : "memory");
                    wv[0 + 0] = __uint_as_float(_ldv8_3_0);
                    wv[0 + 1] = __uint_as_float(_ldv8_3_1);
                    wv[0 + 2] = __uint_as_float(_ldv8_3_2);
                    wv[0 + 3] = __uint_as_float(_ldv8_3_3);
                    wv[0 + 4] = __uint_as_float(_ldv8_3_4);
                    wv[0 + 5] = __uint_as_float(_ldv8_3_5);
                    wv[0 + 6] = __uint_as_float(_ldv8_3_6);
                    wv[0 + 7] = __uint_as_float(_ldv8_3_7);
                }
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    ak[e] = ak[e] + wk[e];
                    av[e] = av[e] + wv[e];
                }
            }
            {
                unsigned _stv8_4_0 = __float_as_uint(ak[0 + 0]);
                unsigned _stv8_4_1 = __float_as_uint(ak[0 + 1]);
                unsigned _stv8_4_2 = __float_as_uint(ak[0 + 2]);
                unsigned _stv8_4_3 = __float_as_uint(ak[0 + 3]);
                unsigned _stv8_4_4 = __float_as_uint(ak[0 + 4]);
                unsigned _stv8_4_5 = __float_as_uint(ak[0 + 5]);
                unsigned _stv8_4_6 = __float_as_uint(ak[0 + 6]);
                unsigned _stv8_4_7 = __float_as_uint(ak[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dk_out + out + (0))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
            }
            {
                unsigned _stv8_5_0 = __float_as_uint(av[0 + 0]);
                unsigned _stv8_5_1 = __float_as_uint(av[0 + 1]);
                unsigned _stv8_5_2 = __float_as_uint(av[0 + 2]);
                unsigned _stv8_5_3 = __float_as_uint(av[0 + 3]);
                unsigned _stv8_5_4 = __float_as_uint(av[0 + 4]);
                unsigned _stv8_5_5 = __float_as_uint(av[0 + 5]);
                unsigned _stv8_5_6 = __float_as_uint(av[0 + 6]);
                unsigned _stv8_5_7 = __float_as_uint(av[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dv_out + out + (0))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
            }
        }
    }
}

} // extern "C"
