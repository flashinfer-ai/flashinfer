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

#include <cstdint>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl {

// Activation type enum for standalone moeActivation kernel
// Note: Matches ActivationType in cutlass_kernels/include/common.h
enum class MoeActivationType {
  Gelu = 0,
  Relu = 1,
  Silu = 2,
  Swiglu = 3,
  Geglu = 4,
  Identity = 5,
};

template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf,
                SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit,
                int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
                int32_t const max_num_permuted_tokens, int32_t const hidden_size,
                int32_t const top_k, int32_t const tile_size, bool enable_pdl, cudaStream_t stream);

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output,
                  int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales,
                  int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
                  bool enable_pdl, cudaStream_t stream);

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit,
                     int32_t const* expanded_idx_to_permuted_idx,
                     int32_t const* permuted_idx_to_expanded_idx,
                     int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens,
                     int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
                     bool enable_pdl, cudaStream_t stream);

// ============================== Activation Kernels ==============================

/**
 * @brief Apply activation function to MoE intermediate outputs.
 *
 * For GLU activations (Swiglu, Geglu), input shape is (num_tokens, 2 * interm_size)
 * where first half is linear projection and second half is gate.
 * Output shape is (num_tokens, interm_size).
 *
 * For non-GLU activations (Gelu, Relu, Silu, Identity), input and output shape
 * are both (num_tokens, interm_size).
 *
 * @param input Input tensor
 * @param output Output tensor (same dtype as input for non-FP4 output)
 * @param tile_idx_to_mn_limit Valid token count per tile
 * @param num_non_exiting_tiles Number of valid tiles (scalar on device)
 * @param activation_type Type of activation to apply
 * @param max_num_permuted_tokens Maximum number of permuted tokens
 * @param interm_size Intermediate size (output hidden dimension)
 * @param tile_size Tile size for scheduling
 * @param enable_pdl Enable Programmatic Dependent Launch
 * @param stream CUDA stream
 */
template <typename InputType>
void moeActivation(InputType const* input, InputType* output, int32_t const* tile_idx_to_mn_limit,
                   int32_t const* num_non_exiting_tiles, MoeActivationType activation_type,
                   int32_t const max_num_permuted_tokens, int32_t const interm_size,
                   int32_t const tile_size, bool enable_pdl, cudaStream_t stream);

/**
 * @brief Fused activation with NVFP4 dynamic quantization.
 *
 * Combines activation function with per-block NVFP4 quantization in a single kernel pass.
 * Output is packed FP4 with swizzled scale factors.
 *
 * @param input Input tensor (bf16/fp16)
 * @param output Output tensor (packed FP4, uint8)
 * @param global_sf Global scale factor for quantization
 * @param output_sf Per-block scale factors (FP8 E4M3, swizzled layout)
 * @param tile_idx_to_mn_limit Valid token count per tile
 * @param num_non_exiting_tiles Number of valid tiles
 * @param activation_type Type of activation to apply
 * @param max_num_permuted_tokens Maximum number of permuted tokens
 * @param interm_size Intermediate size
 * @param tile_size Tile size for scheduling
 * @param enable_pdl Enable Programmatic Dependent Launch
 * @param stream CUDA stream
 */
template <typename InputType, typename OutputType, typename SFType>
void moeActivationQuantize(InputType const* input, OutputType* output, float const* global_sf,
                           SFType* output_sf, int32_t const* tile_idx_to_mn_limit,
                           int32_t const* num_non_exiting_tiles, MoeActivationType activation_type,
                           int32_t const max_num_permuted_tokens, int32_t const interm_size,
                           int32_t const tile_size, bool enable_pdl, cudaStream_t stream);

}  // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
