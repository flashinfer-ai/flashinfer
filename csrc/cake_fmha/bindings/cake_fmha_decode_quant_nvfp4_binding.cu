/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_decode_quant_nvfp4(CakeFmhaTensorMap const* Qt, CakeFmhaTensorMap const* Kp, CakeFmhaTensorMap const* Vp, CakeFmhaTensorMap const* Ksf, CakeFmhaTensorMap const* Vsf, uint8_t* O, int* page_table, int* seq_lens_kv, float* bmm1_scale_ptr, float* bmm2_scale_ptr, float* partial_O, float* partial_max, float* partial_sum, int pt_batch_stride, int pt_v_offset, int bmm1_is_log2, int num_splits, int blocks_per_split);

extern "C" cudaError_t cake_fmha_launch_decode_quant_nvfp4(
    CakeFmhaTensorMap const* Qt,
    CakeFmhaTensorMap const* Kp,
    CakeFmhaTensorMap const* Vp,
    CakeFmhaTensorMap const* Ksf,
    CakeFmhaTensorMap const* Vsf,
    uint8_t* O,
    int* page_table,
    int* seq_lens_kv,
    float* bmm1_scale_ptr,
    float* bmm2_scale_ptr,
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    int pt_batch_stride,
    int pt_v_offset,
    int bmm1_is_log2,
    int num_splits,
    int blocks_per_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_quant_nvfp4),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        147456);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Qt)),
        const_cast<void*>(reinterpret_cast<const void*>(&Kp)),
        const_cast<void*>(reinterpret_cast<const void*>(&Vp)),
        const_cast<void*>(reinterpret_cast<const void*>(&Ksf)),
        const_cast<void*>(reinterpret_cast<const void*>(&Vsf)),
        const_cast<void*>(reinterpret_cast<const void*>(&O)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_kv)),
        const_cast<void*>(reinterpret_cast<const void*>(&bmm1_scale_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&bmm2_scale_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_O)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_max)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_sum)),
        const_cast<void*>(reinterpret_cast<const void*>(&pt_batch_stride)),
        const_cast<void*>(reinterpret_cast<const void*>(&pt_v_offset)),
        const_cast<void*>(reinterpret_cast<const void*>(&bmm1_is_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_splits)),
        const_cast<void*>(reinterpret_cast<const void*>(&blocks_per_split))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_quant_nvfp4),
        dim3(grid_x, grid_y, grid_z),
        dim3(512, 1, 1),
        kernel_args,
        147456,
        stream);
}
