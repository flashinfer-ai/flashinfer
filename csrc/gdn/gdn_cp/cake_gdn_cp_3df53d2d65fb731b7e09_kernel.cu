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

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_gdn_cp_3df53d2d65fb731b7e09(float* __restrict__ packed, int* __restrict__ state_indices, float* __restrict__ output, long long pool_stride0, long long pool_stride1, long long pool_stride2, long long pool_stride3, int num_heads, long long total_values, int use_indices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    long long linear = (long long)blockIdx.x * 256 + (long long)tid;
    if (linear < total_values) {
        long long values_per_seq = (long long)num_heads * 128 * 128;
        int seq_idx = (int)(linear / values_per_seq);
        long long inner = linear % values_per_seq;
        long long head_idx = inner / 16384;
        long long matrix_inner = inner % 16384;
        long long row_idx = matrix_inner / 128;
        long long col_idx = matrix_inner % 128;
        long long pool_row = (long long)seq_idx;
        if (use_indices != 0) {
            pool_row = (long long)state_indices[seq_idx];
        }
        long long output_index = pool_row * pool_stride0 + head_idx * pool_stride1 + row_idx * pool_stride2 + col_idx * pool_stride3;
        output[output_index] = packed[linear];
    }
}

} // extern "C"
