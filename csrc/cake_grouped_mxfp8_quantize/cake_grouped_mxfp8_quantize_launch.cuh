/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::cake_grouped_mxfp8_quantize {

constexpr int32_t kThreads = 128;
constexpr int32_t kQuantBlock = 32;
constexpr int32_t kScaleTileM = 128;
constexpr int32_t kScaleTileK = 128;

inline cudaError_t Launch(const void* input, const int32_t* mask, void* quantized,
                          uint8_t* scales, int32_t batch, int32_t rows, int32_t columns,
                          int32_t padded_columns, int32_t padded_m_tiles,
                          int32_t padded_k_tiles, int32_t blocks_per_row,
                          uint64_t total_tasks, cudaStream_t stream) {
  if (batch == 0 || rows == 0 || blocks_per_row == 0) {
    return cudaSuccess;
  }
  const dim3 grid((static_cast<uint32_t>(blocks_per_row) + kThreads - 1) / kThreads,
                  static_cast<uint32_t>(rows), static_cast<uint32_t>(batch));
  const dim3 block(kThreads, 1, 1);

  const void* input_arg = input;
  const int32_t* mask_arg = mask;
  void* quantized_arg = quantized;
  uint8_t* scales_arg = scales;
  void* args[] = {&input_arg,
                  &mask_arg,
                  &quantized_arg,
                  &scales_arg,
                  &rows,
                  &columns,
                  &padded_columns,
                  &padded_m_tiles,
                  &padded_k_tiles,
                  &blocks_per_row,
                  &total_tasks};
  return cudaLaunchKernel(reinterpret_cast<const void*>(CAKE_GROUPED_MXFP8_KERNEL), grid, block,
                          args, 0, stream);
}

}  // namespace flashinfer::cake_grouped_mxfp8_quantize
