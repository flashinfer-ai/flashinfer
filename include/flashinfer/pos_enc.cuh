/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

/*
 * Positional Encoding for Attention Kernels
 * ==========================================
 *
 * This header contains types and helper functions for positional encoding
 * used INLINE within attention kernels (fused attention+positional encoding):
 *
 * - PosEncodingMode: Enum for positional encoding types (None, RoPE, ALiBi)
 * - get_alibi_slope: ALiBi slope computation
 * - vec_apply_llama_rope*: Inline RoPE helper functions for attention kernels
 *
 * For the STANDALONE RoPE API (apply_rope, apply_llama31_rope, etc.),
 * see flashinfer/rope/pos_enc_kernels.cuh for the full kernel implementations.
 */

#include <cstdint>
#include <string>

#include "math.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief Enumeration of positional encoding modes for attention.
 *
 * Used as a template parameter in attention kernels to control
 * what type of positional encoding is applied inline during attention.
 */
enum class PosEncodingMode {
  kNone = 0U,       ///< No positional encoding
  kRoPELlama = 1U,  ///< Llama-style Rotary Positional Embeddings
  kALiBi = 2U       ///< Attention with Linear Biases
};

/*!
 * \brief Convert PosEncodingMode to human-readable string.
 */
inline std::string PosEncodingModeToString(const PosEncodingMode& pos_encoding_mode) {
  switch (pos_encoding_mode) {
    case PosEncodingMode::kNone:
      return "None";
    case PosEncodingMode::kRoPELlama:
      return "Llama";
    case PosEncodingMode::kALiBi:
      return "ALiBi";
    default:
      return "Unknown";
  }
}

/*!
 * \brief Compute ALiBi slope for a given head index.
 */
__device__ __forceinline__ float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
}

/*
 * Inline RoPE Helper Functions
 * ============================
 * These are used by attention kernels to apply RoPE inline during attention computation.
 */

/*!
 * \brief Apply RoPE to x[0:head_dim] (non-interleaved mode).
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] =
          vec[i] * cos +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE with precomputed cos/sin (non-interleaved mode).
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i] +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE (interleaved mode).
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE with precomputed cos/sin (interleaved mode).
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE with precomputed cos/sin, reusing first half (interleaved mode).
 *
 * In interleaved mode with cos_sin_cache, we only use the first half of cos/sin.
 * This version uses cos[i/2] and sin[i/2] to reuse values for paired elements.
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size>
vec_apply_llama_rope_cos_sin_interleave_reuse_half(const T* x, const vec_t<float, vec_size>& cos,
                                                   const vec_t<float, vec_size>& sin,
                                                   const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i / 2] +
               ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

/*!
 * \brief Scale and store a partial chunk (for RoPE+quantization kernels).
 */
template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void scale_store_partial_chunk(const DType* in_ptr, QuantType* out_ptr,
                                                          uint32_t lane_elem_offset,
                                                          uint32_t chunk_valid, float scale) {
  if (chunk_valid == 0 || lane_elem_offset >= chunk_valid) {
    return;
  }
  vec_t<float, vec_size> vec;
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_load(in_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp.cast_load(in_ptr + elem_idx);
        vec[i] = tmp[0];
      } else {
        vec[i] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    vec[i] = vec[i] * scale;
  }
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_store(out_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp[0] = vec[i];
        tmp.cast_store(out_ptr + elem_idx);
      }
    }
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
