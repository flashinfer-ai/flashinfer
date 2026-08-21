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
kernel_flashinfer_mnnvl_moe_alltoall_stage_combine(uint8_t* __restrict__ payload, uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long destination_offset, unsigned long long payload_bytes, int ep_rank, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (bid == 0) {
        if (tid == 0) {
            unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
            unsigned long long flag_index = ((unsigned long long)ep_rank * workspace_stride_bytes + flag_val_offset) / 4;
            workspace_u32[flag_index] = workspace_u32[flag_index] + 1;
        }
    }
    unsigned long long linear_thread = (unsigned long long)bid * 256 + (unsigned long long)tid;
    unsigned long long thread_stride = (unsigned long long)num_bids * 256;
    #pragma unroll 1
    for (unsigned long long byte_index = linear_thread; byte_index < payload_bytes; byte_index += thread_stride) {
        workspace[destination_offset + byte_index] = payload[byte_index];
    }
}

} // extern "C"
