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
#define SMEM_HISTOGRAM_OFF 0
#define SMEM_HISTOGRAM_STAGE_BYTES 64
#define SMEM_HISTOGRAM_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_mok_count_4_4(int* __restrict__ topk_all, int* __restrict__ counts, int local_tokens, int top_k, int rank)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* histogram = reinterpret_cast<int*>(smem_raw + 0);
    const int histogram_addr = smem + 0;

    // === Task calls (dependency order) ===
    int rank_stride = local_tokens * top_k;
    int route_count = 4 * rank_stride;
    int first_expert = rank * 4;
    int last_expert = first_expert + 4;
    for (int i = tid; i < 16; i += 1024) {
        histogram[i] = 0;
    }
    __syncthreads();
    for (int idx = bid * 1024 + tid; idx < route_count; idx += num_bids * 1024) {
        int peer = idx / rank_stride;
        int expert = topk_all[idx];
        if (expert >= first_expert && expert < last_expert) {
            atomicAdd(&histogram[(expert - first_expert) * 4 + peer], 1);
        }
    }
    __syncthreads();
    for (int i_1 = tid; i_1 < 16; i_1 += 1024) {
        int value = histogram[i_1];
        if (value != 0) {
            atomicAdd(&counts[i_1], value);
        }
    }
}

} // extern "C"
