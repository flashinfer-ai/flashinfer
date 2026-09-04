/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/cake_fmha.h"

extern "C" __global__ void kernel_cake_fmha_context_fp8_hd256(
    CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V,
    uint8_t* O_ptr, int* page_table, int* seq_lens_q, int* seq_lens_kv, int* cu_seq_lens_q,
    float softmax_scale_log2, float output_scale, int total_bh, int max_pages_per_seq,
    uint32_t* dynamic_counter);

extern "C" cudaError_t cake_fmha_launch_context_fp8_hd256(
    CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V,
    uint8_t* O_ptr, int* page_table, int* seq_lens_q, int* seq_lens_kv, int* cu_seq_lens_q,
    float softmax_scale_log2, float output_scale, int total_bh, int max_pages_per_seq,
    uint32_t* dynamic_counter, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    cudaStream_t stream) {
  cudaError_t status =
      cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_cake_fmha_context_fp8_hd256),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 168192);
  if (status != cudaSuccess) {
    return status;
  }
  void* kernel_args[] = {const_cast<void*>(reinterpret_cast<const void*>(&Q)),
                         const_cast<void*>(reinterpret_cast<const void*>(&K)),
                         const_cast<void*>(reinterpret_cast<const void*>(&V)),
                         const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
                         const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
                         const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_q)),
                         const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_kv)),
                         const_cast<void*>(reinterpret_cast<const void*>(&cu_seq_lens_q)),
                         const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
                         const_cast<void*>(reinterpret_cast<const void*>(&output_scale)),
                         const_cast<void*>(reinterpret_cast<const void*>(&total_bh)),
                         const_cast<void*>(reinterpret_cast<const void*>(&max_pages_per_seq)),
                         const_cast<void*>(reinterpret_cast<const void*>(&dynamic_counter))};
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel_cake_fmha_context_fp8_hd256),
                          dim3(grid_x, grid_y, grid_z), dim3(384, 1, 1), kernel_args, 168192,
                          stream);
}
