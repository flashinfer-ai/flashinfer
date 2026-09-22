/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_decode_native_bf16_hd256_smallm_n64_p32(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* partial_O_ptr, float* partial_LSE_ptr, __nv_bfloat16* O_ptr, float* LSE_ptr, uint32_t* counters, int* page_table, int* seq_lens, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads);

extern "C" cudaError_t cake_fmha_launch_decode_native_bf16_hd256_smallm_n64_p32(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __nv_bfloat16* partial_O_ptr,
    float* partial_LSE_ptr,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    uint32_t* counters,
    int* page_table,
    int* seq_lens,
    int max_pages_per_seq,
    float softmax_scale_log2,
    int num_q_heads,
    int num_kv_heads,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_native_bf16_hd256_smallm_n64_p32),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        224512);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Q)),
        const_cast<void*>(reinterpret_cast<const void*>(&K)),
        const_cast<void*>(reinterpret_cast<const void*>(&V)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&counters)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_pages_per_seq)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_q_heads)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_kv_heads))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_native_bf16_hd256_smallm_n64_p32),
        dim3(grid_x, grid_y, grid_z),
        dim3(512, 1, 1),
        kernel_args,
        224512,
        stream);
}
