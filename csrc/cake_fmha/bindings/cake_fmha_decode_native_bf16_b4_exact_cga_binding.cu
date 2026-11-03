/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_decode_native_bf16(CakeFmhaTensorMap const* Qt, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* O_ptr, float* LSE_ptr, int* page_table, int* causal_seqlens_kv_global, float* scale_log2_ptr, float* sinks_ptr, int max_pages_per_seq, int max_local_seq_len, float softmax_scale_log2, int window_left, int num_q_heads, int num_kv_heads, int batch_size);

extern "C" cudaError_t cake_fmha_launch_decode_native_bf16(
    CakeFmhaTensorMap const* Qt,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    int* page_table,
    int* causal_seqlens_kv_global,
    float* scale_log2_ptr,
    float* sinks_ptr,
    int max_pages_per_seq,
    int max_local_seq_len,
    float softmax_scale_log2,
    int window_left,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_native_bf16),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        175360);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Qt)),
        const_cast<void*>(reinterpret_cast<const void*>(&K)),
        const_cast<void*>(reinterpret_cast<const void*>(&V)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
        const_cast<void*>(reinterpret_cast<const void*>(&causal_seqlens_kv_global)),
        const_cast<void*>(reinterpret_cast<const void*>(&scale_log2_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&sinks_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_pages_per_seq)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_local_seq_len)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&window_left)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_q_heads)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_kv_heads)),
        const_cast<void*>(reinterpret_cast<const void*>(&batch_size))
    };
    cudaLaunchAttribute cluster_attribute = {};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim.x = 2;
    cluster_attribute.val.clusterDim.y = 1;
    cluster_attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, grid_y, grid_z);
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 175360;
    config.stream = stream;
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelExC(
        &config,
        reinterpret_cast<const void*>(kernel_cake_fmha_decode_native_bf16),
        kernel_args);
}
