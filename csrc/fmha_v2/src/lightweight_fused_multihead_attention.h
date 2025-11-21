/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fused_multihead_attention.h>

#include <memory>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lightweight_fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef USE_CUBIN_LOADING
// Lightweight wrapper for specific FP8 flash attention kernel
// This function is only available when linking against .cu.o object files
void run_lightweight_flash_attention_e4m3_fp32(
    const bert::Fused_multihead_attention_params_v2& params,
    const bert::Fused_multihead_attention_launch_params& launch_params, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Class for loading and running kernels from cubin files
class CubinFlashAttention {
 public:
  // Constructor loads the cubin file
  CubinFlashAttention(const std::string& cubin_path);

  // Destructor cleans up
  ~CubinFlashAttention();

  // Run the flash attention kernel
  void run(const bert::Fused_multihead_attention_params_v2& params,
           const bert::Fused_multihead_attention_launch_params& launch_params, cudaStream_t stream);

 private:
  CUmodule module_;
  CUfunction kernel_nl_tiled_;
  CUfunction kernel_nl_tiled_causal_;
  bool initialized_;

  // Disable copy
  CubinFlashAttention(const CubinFlashAttention&) = delete;
  CubinFlashAttention& operator=(const CubinFlashAttention&) = delete;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace lightweight_fmha

////////////////////////////////////////////////////////////////////////////////////////////////////
