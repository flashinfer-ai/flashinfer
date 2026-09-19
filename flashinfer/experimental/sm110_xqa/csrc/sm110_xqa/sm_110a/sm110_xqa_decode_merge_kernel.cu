/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
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
static_assert(sizeof(uint64_t) == 8, "Sm110Xqa requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Sm110XqaTensorMap { uint64_t opaque[16]; };
struct __align__(64) Sm110XqaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110XqaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Sm110XqaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) Sm110XqaTensorMapPack { Sm110XqaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(Sm110XqaTensorMap) >= alignof(CUtensorMap), "Sm110XqaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define SM110_XQA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_sm110_xqa_decode_merge(const float* __restrict__ partial, const float* __restrict__ statistics, __half* __restrict__ output, unsigned int partitions)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    unsigned int row = blockIdx.x;
    float maximum = -SM110_XQA_INF;
    #pragma unroll 1
    for (unsigned int partition = 0; partition < partitions; partition++) {
        float candidate = statistics[(row * partitions + partition) * 2];
        float _max_0 = max_noftz(maximum, candidate);
        maximum = _max_0;
    }
    float safe_max = ((maximum == -SM110_XQA_INF) ? 0.0f : maximum);
    float numerator = 0.0f;
    float denominator = 0.0f;
    #pragma unroll 1
    for (unsigned int partition_1 = 0; partition_1 < partitions; partition_1++) {
        unsigned int partial_row = row * partitions + partition_1;
        float partition_max = statistics[partial_row * 2];
        float partition_sum = statistics[partial_row * 2 + 1];
        float _exp2_0 = approx_exp2(partition_max - safe_max);
        float weight = _exp2_0;
        float value = partial[partial_row * 128 + (unsigned int)tid];
        float _fma_0 = __fmaf_rn(value, weight, numerator);
        numerator = _fma_0;
        float _fma_1 = __fmaf_rn(partition_sum, weight, denominator);
        denominator = _fma_1;
    }
    float _rcp_0 = __frcp_rn(denominator);
    float inverse = ((denominator > 0.0f) ? _rcp_0 : 0.0f);
    output[row * 128 + (unsigned int)tid] = numerator * inverse;
}

} // extern "C"
