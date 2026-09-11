/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_context_nvfp4_dequant(uint8_t* K_packed, uint8_t* V_packed, uint8_t* K_scales, uint8_t* V_scales, uint8_t* K_output, uint8_t* V_output, int total_groups, int output_page_stride);

extern "C" cudaError_t cake_fmha_launch_context_nvfp4_dequant(
    uint8_t* K_packed,
    uint8_t* V_packed,
    uint8_t* K_scales,
    uint8_t* V_scales,
    uint8_t* K_output,
    uint8_t* V_output,
    int total_groups,
    int output_page_stride,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_nvfp4_dequant),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        0);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&K_packed)),
        const_cast<void*>(reinterpret_cast<const void*>(&V_packed)),
        const_cast<void*>(reinterpret_cast<const void*>(&K_scales)),
        const_cast<void*>(reinterpret_cast<const void*>(&V_scales)),
        const_cast<void*>(reinterpret_cast<const void*>(&K_output)),
        const_cast<void*>(reinterpret_cast<const void*>(&V_output)),
        const_cast<void*>(reinterpret_cast<const void*>(&total_groups)),
        const_cast<void*>(reinterpret_cast<const void*>(&output_page_stride))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_nvfp4_dequant),
        dim3(grid_x, grid_y, grid_z),
        dim3(256, 1, 1),
        kernel_args,
        0,
        stream);
}
