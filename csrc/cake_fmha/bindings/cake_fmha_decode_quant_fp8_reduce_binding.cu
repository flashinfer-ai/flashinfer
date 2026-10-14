/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_decode_quant_fp8_reduce(float* partial_O, float* partial_max, float* partial_sum, uint8_t* O, float* bmm2_scale_ptr, int num_split);

extern "C" cudaError_t cake_fmha_launch_decode_quant_fp8_reduce(
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    uint8_t* O,
    float* bmm2_scale_ptr,
    int num_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_quant_fp8_reduce),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        0);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&partial_O)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_max)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_sum)),
        const_cast<void*>(reinterpret_cast<const void*>(&O)),
        const_cast<void*>(reinterpret_cast<const void*>(&bmm2_scale_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_split))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_quant_fp8_reduce),
        dim3(grid_x, grid_y, grid_z),
        dim3(32, 1, 1),
        kernel_args,
        0,
        stream);
}
