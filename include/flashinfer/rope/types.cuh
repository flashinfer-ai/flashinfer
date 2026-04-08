/*
 * Copyright (c) 2023-2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ROPE_KERNELS_CUDA_TYPES_CUH_
#define FLASHINFER_ROPE_KERNELS_CUDA_TYPES_CUH_

/*
 * Types for Standalone RoPE Kernels
 * ==================================
 *
 * This header contains type definitions specific to the standalone RoPE API
 * kernels (apply_rope, rope_quantize, etc.):
 *
 * - RopeQuantizeAppendPagedKVCacheParams: Parameters for combined RoPE +
 *   quantization + KV cache append operations.
 *
 * For attention-level positional encoding types (PosEncodingMode, etc.),
 * see include/flashinfer/pos_enc.cuh.
 */

#include <cstdint>

namespace flashinfer {

/*!
 * \brief Parameters for the combined RoPE + quantization + paged KV cache append kernel.
 *
 * Contains all necessary strides, dimensions, and quantization scales for
 * processing Q, K, V tensors with RoPE and FP8 quantization.
 */
struct RopeQuantizeAppendPagedKVCacheParams {
  uint32_t nnz;                ///< Number of non-zero elements (total tokens)
  uint32_t num_qo_heads;       ///< Number of query/output heads
  uint32_t num_kv_heads;       ///< Number of key/value heads
  uint32_t rope_dim;           ///< Dimension of rotary embeddings
  uint32_t no_rope_dim;        ///< Dimension without rotary embeddings
  size_t q_rope_in_stride_n;   ///< Q RoPE input stride (token dimension)
  size_t q_rope_in_stride_h;   ///< Q RoPE input stride (head dimension)
  size_t q_nope_in_stride_n;   ///< Q non-RoPE input stride (token dimension)
  size_t q_nope_in_stride_h;   ///< Q non-RoPE input stride (head dimension)
  size_t q_rope_out_stride_n;  ///< Q RoPE output stride (token dimension)
  size_t q_rope_out_stride_h;  ///< Q RoPE output stride (head dimension)
  size_t q_nope_out_stride_n;  ///< Q non-RoPE output stride (token dimension)
  size_t q_nope_out_stride_h;  ///< Q non-RoPE output stride (head dimension)
  size_t k_rope_in_stride;     ///< K RoPE input stride (token dimension)
  size_t k_rope_in_stride_h;   ///< K RoPE input stride (head dimension)
  size_t k_nope_in_stride;     ///< K non-RoPE input stride (token dimension)
  size_t k_nope_in_stride_h;   ///< K non-RoPE input stride (head dimension)
  size_t v_in_stride;          ///< V input stride (token dimension)
  size_t v_in_stride_h;        ///< V input stride (head dimension)
  float quant_scale_q;         ///< Quantization scale for Q
  float quant_scale_kv;        ///< Quantization scale for K/V
};

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_KERNELS_CUDA_TYPES_CUH_
