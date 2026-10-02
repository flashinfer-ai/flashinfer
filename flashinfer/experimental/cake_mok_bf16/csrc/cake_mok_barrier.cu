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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1) void
kernel_cake_mok_barrier(unsigned int* __restrict__ local_counter, unsigned int* __restrict__ target_counter, unsigned long long multicast_address, unsigned int ep_size)
{
    const int tid = threadIdx.x;
    const int warp = 0;
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    unsigned int _atomic_old_0 = atomicAdd(target_counter, ep_size);
    unsigned int target = _atomic_old_0 + ep_size;
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(multicast_address)), "r"((int)(1)) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
    uint32_t _relaxed_ld_0;
    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(local_counter) : "memory");
    unsigned int value = _relaxed_ld_0;
    while (value < target) {
        uint32_t _relaxed_ld_1;
        asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_relaxed_ld_1) : "l"(local_counter) : "memory");
        value = _relaxed_ld_1;
    }
    asm volatile("fence.acquire.sys;" ::: "memory");
}

} // extern "C"
