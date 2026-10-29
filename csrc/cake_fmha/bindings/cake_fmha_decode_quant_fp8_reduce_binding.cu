/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_decode_quant_fp8_reduce(float* partial_O, float* partial_max, float* partial_sum, uint8_t* O, float* bmm2_scale_ptr, int num_split, float bmm2_scalar);

extern "C" cudaError_t cake_fmha_launch_decode_quant_fp8_reduce(
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    uint8_t* O,
    float* bmm2_scale_ptr,
    int num_split,
    float bmm2_scalar,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream,
    bool use_pdl) {
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
        const_cast<void*>(reinterpret_cast<const void*>(&num_split)),
        const_cast<void*>(reinterpret_cast<const void*>(&bmm2_scalar))
    };
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, grid_y, grid_z);
    config.blockDim = dim3(32, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    // use_pdl only for a reducer built with USE_PDL: it waits on the decode grid (griddepcontrol.wait) before it
    // reads the partials, so it may launch while that grid drains.
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = use_pdl ? attrs : nullptr;
    config.numAttrs = use_pdl ? 1 : 0;
    return cudaLaunchKernelExC(
        &config, reinterpret_cast<const void*>(kernel_cake_fmha_decode_quant_fp8_reduce), kernel_args);
}
