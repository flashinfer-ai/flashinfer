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

__global__ __launch_bounds__(THREADS) void
kernel_cake_mok_expert_layout(int* __restrict__ counts, int* __restrict__ schedule_rank, int* __restrict__ layout, int experts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    if (warp_1 == 0) {
        int carry = 0;
        #pragma unroll 1
        for (int chunk = 0; chunk < experts; chunk += 64) {
            int lo_e = chunk + lane_1;
            int hi_e = chunk + 32 + lane_1;
            int lo_c = 0;
            int hi_c = 0;
            if (lo_e < experts) {
                lo_c = counts[lo_e];
            }
            if (hi_e < experts) {
                hi_c = counts[hi_e];
            }
            uint32_t _warp_scan_sum_0 = (unsigned int)lo_c;
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_0) : "r"(1));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_0) : "r"(2));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_0) : "r"(4));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_0) : "r"(8));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_0) : "r"(16));
            int lo_s = (int)_warp_scan_sum_0;
            uint32_t _warp_scan_sum_1 = (unsigned int)hi_c;
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_1) : "r"(1));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_1) : "r"(2));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_1) : "r"(4));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_1) : "r"(8));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_1) : "r"(16));
            int hi_s = (int)_warp_scan_sum_1;
            int lo_t = __shfl_sync(0xFFFFFFFF, lo_s, 31);
            int hi_t = __shfl_sync(0xFFFFFFFF, hi_s, 31);
            if (lo_e < experts) {
                layout[experts + lo_e] = carry + lo_s - lo_c;
            }
            if (hi_e < experts) {
                layout[experts + hi_e] = carry + lo_t + hi_s - hi_c;
            }
            carry = carry + lo_t + hi_t;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int expert = tid; expert < experts; expert += 256) {
        int start = layout[experts + expert];
        int padded = counts[expert];
        int real = 0;
        if (padded > 0) {
            int low = padded - 256;
            int high = padded;
            #pragma unroll
            for (int _ = 0; _ < 8; _++) {
                int mid = (low + high) / 2;
                int probe = schedule_rank[start + mid];
                low = ((probe >= 0) ? mid : low);
                high = ((probe >= 0) ? high : mid);
            }
            real = high;
        }
        layout[2 * experts + expert] = real;
        #pragma unroll 1
        for (int block = start / 256; block < (start + padded) / 256; block++) {
            layout[3 * experts + block] = expert;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int expert_1 = tid; expert_1 < experts; expert_1 += 256) {
        layout[expert_1] = counts[expert_1];
    }
}

} // extern "C"
