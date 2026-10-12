/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_dcp_spec_bf16_balanced(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* O_ptr, float* LSE_ptr, int* page_table, int* causal_seqlens_kv_global, float* partial_o, float* partial_stats, uint32_t* tile_counters, uint32_t* queue_counters, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, int batch_size, int q_len, int cp_rank, int cp_world_log2, uint32_t max_items);

extern "C" cudaError_t cake_fmha_launch_dcp_spec_bf16_balanced(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    int* page_table,
    int* causal_seqlens_kv_global,
    float* partial_o,
    float* partial_stats,
    uint32_t* tile_counters,
    uint32_t* queue_counters,
    int max_pages_per_seq,
    float softmax_scale_log2,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    int q_len,
    int cp_rank,
    int cp_world_log2,
    uint32_t max_items,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_dcp_spec_bf16_balanced),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        226304);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Q)),
        const_cast<void*>(reinterpret_cast<const void*>(&K)),
        const_cast<void*>(reinterpret_cast<const void*>(&V)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
        const_cast<void*>(reinterpret_cast<const void*>(&causal_seqlens_kv_global)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_o)),
        const_cast<void*>(reinterpret_cast<const void*>(&partial_stats)),
        const_cast<void*>(reinterpret_cast<const void*>(&tile_counters)),
        const_cast<void*>(reinterpret_cast<const void*>(&queue_counters)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_pages_per_seq)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_q_heads)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_kv_heads)),
        const_cast<void*>(reinterpret_cast<const void*>(&batch_size)),
        const_cast<void*>(reinterpret_cast<const void*>(&q_len)),
        const_cast<void*>(reinterpret_cast<const void*>(&cp_rank)),
        const_cast<void*>(reinterpret_cast<const void*>(&cp_world_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_items))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_dcp_spec_bf16_balanced),
        dim3(grid_x, grid_y, grid_z),
        dim3(384, 1, 1),
        kernel_args,
        226304,
        stream);
}
