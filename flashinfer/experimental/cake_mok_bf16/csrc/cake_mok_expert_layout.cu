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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mok_expert_layout(int* __restrict__ counts, int* __restrict__ schedule_rank, int* __restrict__ layout, int experts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (tid == 0) {
        int offset = 0;
        #pragma unroll 1
        for (int expert = 0; expert < experts; expert++) {
            int count = counts[expert];
            layout[expert] = count;
            layout[experts + expert] = offset;
            offset = offset + count;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int expert_1 = tid; expert_1 < experts; expert_1 += 256) {
        int start = layout[experts + expert_1];
        int real = layout[expert_1];
        #pragma unroll 1
        for (int _ = 0; _ < 256; _++) {
            if (real == 0) {
                break;
            }
            int last_peer = schedule_rank[start + real - 1];
            if (last_peer >= 0) {
                break;
            }
            real = real - 1;
        }
        layout[2 * experts + expert_1] = real;
        #pragma unroll 1
        for (int block = start / 256; block < (start + layout[expert_1]) / 256; block++) {
            layout[3 * experts + block] = expert_1;
        }
    }
}

} // extern "C"
