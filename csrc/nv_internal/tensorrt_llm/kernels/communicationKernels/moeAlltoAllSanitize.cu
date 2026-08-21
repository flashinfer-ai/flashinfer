/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_sanitize_expert_ids(int* __restrict__ expert_ids, int* __restrict__ recv_counters, int ep_size, int max_tokens_per_rank, int top_k, int invalid_id, int enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (enable_pdl != 0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tid_0 = bid * 256 + tid;
    int total_tokens = ep_size * max_tokens_per_rank;
    if (tid_0 < total_tokens) {
        int source_rank = tid_0 / max_tokens_per_rank;
        int token_idx = tid_0 - source_rank * max_tokens_per_rank;
        if (token_idx >= recv_counters[source_rank]) {
            int token_base = tid_0 * top_k;
            #pragma unroll 1
            for (int k = 0; k < top_k; k++) {
                expert_ids[token_base + k] = invalid_id;
            }
        }
    }
}

} // extern "C"
