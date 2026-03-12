/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include "quantization.h"


// Unified launch function template for fused RMSNorm + SiLU
// Handles both standard and NDHWC output layouts via template parameter USE_NDHWC
// USE_NDHWC = false: Standard layout [num_tokens, hidden_dim]
// USE_NDHWC = true: NDHWC layout [N, C, D_total, H, W] with channels_last_3d format
//
// Note: For NDHWC layout, output_stride_C MUST be 1 (channels must be contiguous).
// This is required for vectorized stores and is enforced by runtime assertion.
// Formula matches open source WAN implementation: F.normalize(x, dim=1) * scale * gamma + bias
// where F.normalize(x, dim=1) = x / sqrt(sum(x^2) + eps)
template <bool USE_NDHWC>
void launchFusedRMSNormSiLU(
    void const* input,            // Input tensor [num_tokens, hidden_dim]
    void* output,                 // Output tensor [num_tokens, hidden_dim] or NDHWC layout
    int const num_tokens,         // Number of tokens
    int const hidden_dim,         // Hidden dimension
    int const input_stride_token, // Stride between tokens in input (in elements)
    int const num_sms,            // Number of SMs (from PyTorch cached properties)
    float const eps,              // Epsilon for RMS normalization
    void const* weight,           // RMSNorm weights [hidden_dim]
    float const scale,            // Scale factor (typically sqrt(hidden_dim))
    void const* bias,             // Optional bias [hidden_dim] (can be nullptr)
    // Standard layout parameters (used when USE_NDHWC == false)
    int const output_stride_token, // Stride between tokens in output (in elements)
    // NDHWC layout parameters (used when USE_NDHWC == true)
    int const DHW,                // Depth * Height * Width (precomputed: D * H * W)
    int const HW,                 // Height * Width (precomputed: H * W)
    int const W,                  // Width dimension
    int const output_stride_N,    // Stride for N dimension in output (in elements)
    int const output_stride_D,    // Stride for D dimension in output (in elements)
    int const output_stride_H,    // Stride for H dimension in output (in elements)
    int const output_stride_W,    // Stride for W dimension in output (in elements)
    int const output_D_offset,    // Offset in D dimension where to start writing
    cudaStream_t stream);         // CUDA stream

// Overloaded version with quantization support
template <bool USE_NDHWC>
void launchFusedRMSNormSiLU(
    void const* input,
    void* output,
    int const num_tokens,
    int const hidden_dim,
    int const input_stride_token,
    int const num_sms,
    float const eps,
    void const* weight,
    float const scale,
    void const* bias,
    int const output_stride_token,
    int const DHW,
    int const HW,
    int const W,
    int const output_stride_N,
    int const output_stride_D,
    int const output_stride_H,
    int const output_stride_W,
    int const output_D_offset,
    QuantConfig const& quant_config,  // Quantization configuration
    cudaStream_t stream);

