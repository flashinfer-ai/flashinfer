/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_context_nvfp4(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* Kp, CakeFmhaTensorMap const* Vp, CakeFmhaTensorMap const* Ksf, CakeFmhaTensorMap const* Vsf, uint8_t* O_ptr, float* LSE_ptr, float* sinks, int* page_table_k, int* page_table_v, int* seq_lens_q, int* seq_lens_kv, int* cu_seq_lens_q, float softmax_scale_log2, float output_scale, int total_bh, int page_row_stride, int num_ctas, uint32_t* dynamic_counter);

extern "C" cudaError_t cake_fmha_launch_context_nvfp4(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* Kp,
    CakeFmhaTensorMap const* Vp,
    CakeFmhaTensorMap const* Ksf,
    CakeFmhaTensorMap const* Vsf,
    uint8_t* O_ptr,
    float* LSE_ptr,
    float* sinks,
    int* page_table_k,
    int* page_table_v,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    float output_scale,
    int total_bh,
    int page_row_stride,
    int num_ctas,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_nvfp4),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        104960);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Q)),
        const_cast<void*>(reinterpret_cast<const void*>(&Kp)),
        const_cast<void*>(reinterpret_cast<const void*>(&Vp)),
        const_cast<void*>(reinterpret_cast<const void*>(&Ksf)),
        const_cast<void*>(reinterpret_cast<const void*>(&Vsf)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&sinks)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table_k)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table_v)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_kv)),
        const_cast<void*>(reinterpret_cast<const void*>(&cu_seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&output_scale)),
        const_cast<void*>(reinterpret_cast<const void*>(&total_bh)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_row_stride)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_ctas)),
        const_cast<void*>(reinterpret_cast<const void*>(&dynamic_counter))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_nvfp4),
        dim3(grid_x, grid_y, grid_z),
        dim3(512, 1, 1),
        kernel_args,
        104960,
        stream);
}
