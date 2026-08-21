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
#define THREADS 64

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

extern "C" {

__global__ __launch_bounds__(64) void
kernel_flashinfer_mnnvl_moe_alltoall_publish_combine(uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long completion_flags_offset, int ep_rank, int ep_size, bool enable_pdl, bool enable_rank_mask, unsigned long long active_rank_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    unsigned long long expected_index = (local_workspace_base + flag_val_offset) / 4;
    unsigned int expected_value = workspace_u32[expected_index];
    asm volatile("fence.release.sys;" ::: "memory");
    if (tid < ep_size) {
        int target_rank = tid;
        int target_is_active = 1;
        if (enable_rank_mask) {
            unsigned long long rank_mask_bit = active_rank_mask >> (unsigned long long)target_rank & 1;
            target_is_active = ((rank_mask_bit != 0) ? 1 : 0);
        }
        if (target_is_active != 0) {
            unsigned long long remote_flag_index = ((unsigned long long)target_rank * workspace_stride_bytes + completion_flags_offset) / 4 + (unsigned long long)ep_rank;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(workspace_u32) + (remote_flag_index))), "r"(static_cast<unsigned int>(expected_value)) : "memory");
            unsigned long long local_flag_index = (local_workspace_base + completion_flags_offset) / 4 + (unsigned long long)target_rank;
            {
                unsigned int* _sre_ptr_0 = (reinterpret_cast<unsigned int*>(workspace_u32) + (local_flag_index));
                const unsigned int _sre_expected_0 = static_cast<unsigned int>(expected_value);
                const unsigned long long _sre_start_0 = clock64();
                bool _sre_matched_0 = false;
                do {
                    unsigned int _sre_value_0;
                    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_0) : "l"(_sre_ptr_0) : "memory");
                    _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
                } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <= static_cast<unsigned long long>(600000000000)));
                if (__builtin_expect(!_sre_matched_0, 0)) {
                    asm volatile("trap;");
                    return;
                }
            }
        }
    }
    __syncthreads();
    asm volatile("fence.acquire.sys;" ::: "memory");
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"
