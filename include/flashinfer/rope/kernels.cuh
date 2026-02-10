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
#ifndef FLASHINFER_ROPE_KERNELS_CUDA_KERNELS_CUH_
#define FLASHINFER_ROPE_KERNELS_CUDA_KERNELS_CUH_

/*
 * CUDA Kernel Implementations for RoPE (Rotary Positional Embeddings)
 * ====================================================================
 *
 * This header contains the CUDA kernel implementations (device code) for RoPE:
 *
 * Standard RoPE Kernels (using indptr/offsets for batched ragged tensors):
 * - BatchQKApplyRotaryKernel: Apply RoPE using batch indptr and position offsets
 *
 * Position ID RoPE Kernels (using explicit position IDs):
 * - BatchQKApplyRotaryPosIdsKernel: Sequential heads, one token per thread-group
 * - BatchQKApplyRotaryPosIdsHeadParallelismKernel: Parallel heads for small workloads
 *
 * Cos/Sin Cache RoPE Kernels (using precomputed cos/sin tables):
 * - BatchQKApplyRotaryPosIdsCosSinCacheKernel: Sequential heads with cos/sin cache
 * - BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel: Parallel heads with cache
 *
 * Combined RoPE + Quantization Kernels:
 * - RopeQuantizeKernel: RoPE + FP8 quantization (for MLA architecture)
 * - RopeQuantizeAppendPagedKVCacheKernel: RoPE + quantize + append to paged cache
 *
 * Kernel Selection Strategy:
 * - For large workloads (many tokens): Use sequential-heads kernels
 * - For small workloads (few tokens): Use head-parallelism kernels
 * The launcher functions automatically select based on GPU occupancy.
 */

#include <cstdint>
#include <type_traits>

#include "flashinfer/layout.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/rope/types.cuh"
#include "flashinfer/vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief Kernel to apply RoPE with cos/sin cache and head parallelism.
 *
 * Each thread block processes one token, with the y-dimension of the grid
 * iterating over heads. Suitable for small token counts where we want
 * to maximize parallelism across heads.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam head_dim Head dimension (compile-time constant)
 * \tparam vec_size Vector size for loads/stores
 * \tparam bdx Threads per head dimension
 * \tparam DType Data type (e.g., half, bfloat16)
 * \tparam IdType Index type for position IDs
 */
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    // Load cos/sin from cache based on interleave mode:
    // interleave: cos = cache[pos][tx * vec_size // 2], sin = cache[pos][rotary_dim/2 + tx *
    // vec_size // 2] non-interleave: cos = cache[pos][(tx * vec_size) % half_rotary_dim], sin =
    // cache[pos][half_rotary_dim + ...]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/*!
 * \brief Kernel to apply RoPE with cos/sin cache (sequential heads).
 *
 * Each thread block processes one token, iterating over all heads sequentially.
 * More efficient for large token counts as it reuses loaded cos/sin values
 * across all heads.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam head_dim Head dimension
 * \tparam vec_size Vector size for loads/stores
 * \tparam bdx Threads per head dimension
 * \tparam DType Data type
 * \tparam IdType Index type for position IDs
 */
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // Load cos/sin values (same indexing as head parallelism kernel)
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    // Process all Q heads sequentially (don't unroll - head count might be large)
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

    // Process all K heads sequentially
#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/*!
 * \brief Combined RoPE + quantization kernel.
 *
 * Applies RoPE to Q and K tensors while simultaneously quantizing to FP8.
 * Handles separate rope/nope (non-rotary) slices of the head dimension.
 * Used primarily for MLA (Multi-head Latent Attention) architectures.
 *
 * Block allocation strategy:
 * - Blocks [0, num_qo_heads): Q RoPE processing
 * - Blocks [num_qo_heads, num_qo_heads + num_kv_heads): K RoPE processing
 * - Blocks [num_qo_heads + num_kv_heads, ...): K non-RoPE processing
 * - Remaining blocks: Q non-RoPE processing
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam vec_size Vector size for loads/stores
 * \tparam bdx Threads per dimension
 * \tparam DType Input data type
 * \tparam IdType Index type for position IDs
 * \tparam QuantType Output quantized type (e.g., fp8_e4m3)
 */
template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType,
          typename QuantType>
__global__ void RopeQuantizeKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out,
    float* __restrict__ cos_sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h, size_t q_nope_in_stride_n,
    size_t q_nope_in_stride_h, size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h, size_t k_rope_in_stride,
    size_t k_rope_in_stride_h, size_t k_nope_in_stride, size_t k_nope_in_stride_h,
    size_t k_rope_out_stride, size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  // Calculate block allocation boundaries
  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rope_dim = rope_dim / 2;
    // Load cos/sin for RoPE blocks only
    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // Q RoPE processing
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // K RoPE processing
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_rope_in_stride, k_rope_in_stride_h);
      QuantType* k_rope_out_ptr =
          k_rope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_rope_out_stride,
                                            k_rope_out_stride_h);

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }
      k_rope_vec.cast_store(k_rope_out_ptr + tx * vec_size);

    } else if (by < k_nope_end) {
      // K Non-RoPE processing (just scale, no rotation)
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_nope_in_stride, k_nope_in_stride_h);
      QuantType* k_nope_out_ptr =
          k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_nope_out_stride,
                                            k_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          k_nope_in_ptr, k_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_kv);

    } else {
      // Q Non-RoPE processing
      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          q_nope_in_ptr, q_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_q);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/*!
 * \brief Kernel to apply RoPE with position IDs and head parallelism.
 *
 * Each block processes one token with head-parallel execution.
 * Suitable for small token counts where GPU occupancy is a concern.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam head_dim Head dimension
 * \tparam vec_size Vector size
 * \tparam bdx Threads per head dimension
 * \tparam DType Data type
 * \tparam IdType Position ID type
 */
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  // Compute frequency vector (with optional Llama 3.1 smoothing)
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      // Apply Llama 3.1 smoothing if enabled (smooth_a != 0)
      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/*!
 * \brief Kernel to apply RoPE with position IDs (sequential heads).
 *
 * Each block processes one token, iterating through all heads sequentially.
 * More efficient for large token counts.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam head_dim Head dimension
 * \tparam vec_size Vector size
 * \tparam bdx Threads per head dimension
 * \tparam DType Data type
 * \tparam IdType Position ID type
 */
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  // Compute frequency vector
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

    // Process all Q heads sequentially
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

    // Process all K heads sequentially
#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/*!
 * \brief Kernel to apply RoPE with indptr/offsets (for ragged batched tensors).
 *
 * Uses batch indptr array to determine sequence boundaries and offsets
 * array for position computation. Each block handles one (batch, head) pair.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam head_dim Head dimension
 * \tparam vec_size Vector size
 * \tparam bdx Threads per head dimension
 * \tparam DType Data type
 * \tparam IdType Index type
 */
template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  // Compute frequency vector
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // Apply rotary to Q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr =
            q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          q_vec =
              vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {
    // Apply rotary to K
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr =
            k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          k_vec =
              vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}

/*!
 * \brief Combined RoPE + quantization + paged KV cache append kernel.
 *
 * Unified kernel that applies RoPE to Q/K, quantizes to FP8, and appends
 * K/V to the paged KV cache. Supports both MLA and GQA/MHA configurations.
 *
 * \tparam interleave Whether to use interleaved RoPE layout
 * \tparam vec_size Vector size
 * \tparam bdx Threads per dimension
 * \tparam DType Input data type
 * \tparam RoPEIdType Position ID type for RoPE
 * \tparam PagedKVIdType Index type for paged KV cache
 * \tparam QuantType Output quantized type
 * \tparam CacheT Paged cache type (paged_kv_t for GQA/MHA, paged_kv_mla_t for MLA)
 */
template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename RoPEIdType,
          typename PagedKVIdType, typename QuantType, typename CacheT>
__global__ void RopeQuantizeAppendPagedKVCacheKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, DType* v_in,
    QuantType* q_rope_out, QuantType* q_nope_out, CacheT paged_kv_like,
    PagedKVIdType* __restrict__ batch_indices, PagedKVIdType* __restrict__ positions,
    float* __restrict__ cos_sin_cache, RoPEIdType* __restrict__ pos_ids,
    const RopeQuantizeAppendPagedKVCacheParams params) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  // Extract params for readability
  const uint32_t nnz = params.nnz;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t rope_dim = params.rope_dim;
  const uint32_t no_rope_dim = params.no_rope_dim;
  const size_t q_rope_in_stride_n = params.q_rope_in_stride_n;
  const size_t q_rope_in_stride_h = params.q_rope_in_stride_h;
  const size_t q_nope_in_stride_n = params.q_nope_in_stride_n;
  const size_t q_nope_in_stride_h = params.q_nope_in_stride_h;
  const size_t q_rope_out_stride_n = params.q_rope_out_stride_n;
  const size_t q_rope_out_stride_h = params.q_rope_out_stride_h;
  const size_t q_nope_out_stride_n = params.q_nope_out_stride_n;
  const size_t q_nope_out_stride_h = params.q_nope_out_stride_h;
  const size_t k_rope_in_stride = params.k_rope_in_stride;
  const size_t k_rope_in_stride_h = params.k_rope_in_stride_h;
  const size_t k_nope_in_stride = params.k_nope_in_stride;
  const size_t k_nope_in_stride_h = params.k_nope_in_stride_h;
  const size_t v_in_stride = params.v_in_stride;
  const size_t v_in_stride_h = params.v_in_stride_h;
  const float quant_scale_q = params.quant_scale_q;
  const float quant_scale_kv = params.quant_scale_kv;

  // Block allocation boundaries
  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  // Detect MLA vs GQA/MHA from cache type
  constexpr bool IS_MLA = std::is_same<CacheT, paged_kv_mla_t<QuantType, PagedKVIdType>>::value;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const RoPEIdType pos = pos_ids[idx];

    // Compute page location for this token
    uint32_t page_iter, entry_idx;
    paged_kv_like.page_size.divmod(
        paged_kv_like.indptr[batch_indices[idx]] * paged_kv_like.page_size + positions[idx],
        page_iter, entry_idx);

    const int half_rope_dim = rope_dim / 2;
    // Load cos/sin for RoPE blocks only
    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // Q RoPE processing
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // K RoPE processing & cache append
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr;
      if constexpr (IS_MLA) {
        k_rope_in_ptr = k_rope_in + idx * k_rope_in_stride + elem_offset;
      } else {
        k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                         k_rope_in_stride, k_rope_in_stride_h);
      }

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }

      if constexpr (IS_MLA) {
        QuantType* kpe_ptr =
            paged_kv_like.get_kpe_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
        k_rope_vec.cast_store(kpe_ptr);
      } else {
        QuantType* k_ptr = paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx, tx * vec_size);
        k_rope_vec.cast_store(k_ptr);
      }

    } else if (by < k_nope_end) {
      // K Non-RoPE processing & cache append
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* k_nope_in_ptr;
      if constexpr (IS_MLA) {
        k_nope_in_ptr = k_nope_in + idx * k_nope_in_stride + elem_offset;
      } else {
        k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                         k_nope_in_stride, k_nope_in_stride_h);
      }

      vec_t<float, vec_size> k_nope_vec;
      k_nope_vec.cast_load(k_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_nope_vec[i] = k_nope_vec[i] * quant_scale_kv;
      }

      if constexpr (IS_MLA) {
        QuantType* ckv_ptr =
            paged_kv_like.get_ckv_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
        k_nope_vec.cast_store(ckv_ptr);
      } else {
        QuantType* k_ptr = paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx,
                                                   rope_dim + elem_offset + tx * vec_size);
        k_nope_vec.cast_store(k_ptr);
      }

    } else if (by < k_nope_end + (IS_MLA ? 0u : num_kv_heads)) {
      // V processing & cache append (GQA/MHA only)
      if constexpr (!IS_MLA) {
        uint32_t kv_head_idx = by - k_nope_end;
        DType* v_in_ptr =
            v_in + get_elem_offset_impl(idx, kv_head_idx, 0, v_in_stride, v_in_stride_h);
        uint32_t head_dim_total = rope_dim + no_rope_dim;
        uint32_t v_chunks = (head_dim_total + rope_chunk_size - 1) / rope_chunk_size;
#pragma unroll 1
        for (uint32_t j = 0; j < v_chunks; ++j) {
          uint32_t v_elem_offset = j * rope_chunk_size;
          if (v_elem_offset + tx * vec_size < head_dim_total) {
            vec_t<float, vec_size> v_vec;
            v_vec.cast_load(v_in_ptr + v_elem_offset + tx * vec_size);
#pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) {
              v_vec[i] = v_vec[i] * quant_scale_kv;
            }
            QuantType* v_ptr = paged_kv_like.get_v_ptr(page_iter, kv_head_idx, entry_idx,
                                                       v_elem_offset + tx * vec_size);
            v_vec.cast_store(v_ptr);
          }
        }
      }

    } else {
      // Q Non-RoPE processing
      uint32_t q_nope_start = k_nope_end + (IS_MLA ? 0u : num_kv_heads);
      uint32_t q_head_idx = (by - q_nope_start) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - q_nope_start) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      vec_t<float, vec_size> q_nope_vec;
      q_nope_vec.cast_load(q_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_nope_vec[i] = q_nope_vec[i] * quant_scale_q;
      }
      q_nope_out_ptr = q_nope_out_ptr + tx * vec_size;
      q_nope_vec.cast_store(q_nope_out_ptr);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_KERNELS_CUDA_KERNELS_CUH_
