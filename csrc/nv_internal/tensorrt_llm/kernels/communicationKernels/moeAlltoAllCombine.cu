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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, int top_k, bool use_low_precision, bool enable_pdl)
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
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)top_k;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)top_k;
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float result = 0.0f;
            #pragma unroll 1
            for (int route = 0; route < top_k; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    result = result + contribution;
                }
            }
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"
