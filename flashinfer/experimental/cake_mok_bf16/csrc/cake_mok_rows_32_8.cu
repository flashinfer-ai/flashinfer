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
#define SMEM_PEER_COUNTS_OFF 0
#define SMEM_PEER_COUNTS_STAGE_BYTES 128
#define SMEM_PEER_COUNTS_STRIDE 128
#define SMEM_PREFIX_OFF 0
#define SMEM_PREFIX_STAGE_BYTES 128
#define SMEM_PREFIX_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_mok_rows_32_8(int* __restrict__ topk_all, int* __restrict__ counts, int* __restrict__ tokens_per_expert, int* __restrict__ num_tokens, int* __restrict__ schedule_peer_rank, int* __restrict__ schedule_peer_token_idx, int local_tokens, int top_k, int capacity, int rank)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(4) unsigned char smem_static_raw[128];
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
    int* peer_counts = reinterpret_cast<int*>(smem_raw + 0);
    const int peer_counts_addr = smem + 0;
    int* prefix = reinterpret_cast<int*>(smem_static_raw + 0);
    const int prefix_addr = (int)(unsigned long long)__cvta_generic_to_shared(prefix);

    // === Task calls (dependency order) ===
    int rank_stride = local_tokens * top_k;
    int first_expert = rank * 8;
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    if (num_tokens[0] > capacity) {
        asm volatile("trap;" ::: "memory");
    }
    for (int idx = bid; idx < 256; idx += num_bids) {
        int local_expert = idx / 32;
        int peer = idx % 32;
        int expert_base = 0;
        #pragma unroll 1
        for (int e = 0; e < local_expert; e++) {
            expert_base = expert_base + tokens_per_expert[e];
        }
        for (int p = tid; p < 32; p += 1024) {
            peer_counts[p] = counts[local_expert * 32 + p];
        }
        __syncthreads();
        int owned = 0;
        for (int token = tid; token < rank_stride; token += 1024) {
            int expert = topk_all[peer * rank_stride + token];
            owned = owned + ((expert - first_expert == local_expert) ? 1 : 0);
        }
        int inclusive = owned;
        #pragma unroll
        for (int bit = 0; bit < 5; bit++) {
            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1 << bit, 32);
            if (lane_1 >= 1 << bit) {
                inclusive = inclusive + _shfl_up_0;
            }
        }
        if (lane_1 == 31) {
            prefix[warp_1] = inclusive;
        }
        __syncthreads();
        if (warp_1 == 0) {
            int warp_total = prefix[lane_1];
            #pragma unroll
            for (int bit_1 = 0; bit_1 < 5; bit_1++) {
                int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, warp_total, 1 << bit_1, 32);
                if (lane_1 >= 1 << bit_1) {
                    warp_total = warp_total + _shfl_up_1;
                }
            }
            prefix[lane_1] = warp_total;
        }
        __syncthreads();
        int previous = 0;
        if (warp_1 > 0) {
            previous = prefix[warp_1 - 1];
        }
        int ordinal = previous + inclusive - owned;
        for (int token_1 = tid; token_1 < rank_stride; token_1 += 1024) {
            int expert_1 = topk_all[peer * rank_stride + token_1];
            if (expert_1 - first_expert == local_expert) {
                int dst = expert_base;
                #pragma unroll 1
                for (int p_1 = 0; p_1 < 32; p_1++) {
                    int peer_total = peer_counts[p_1];
                    int _min_0 = ((peer_total) < (ordinal) ? (peer_total) : (ordinal));
                    dst = dst + _min_0;
                    if (peer > p_1 && peer_total > ordinal) {
                        dst = dst + 1;
                    }
                }
                schedule_peer_rank[dst] = peer;
                schedule_peer_token_idx[dst] = token_1;
                ordinal = ordinal + 1;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
