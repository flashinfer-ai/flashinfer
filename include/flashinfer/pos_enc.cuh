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

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "layout.cuh"
#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kRoPELlama = 1U,
  // Apply ALiBi bias
  kALiBi = 2U
};

/*!
 * \brief Convert PosEncodingMode to string
 * \param pos_encoding_mode A PosEncodingMode value
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

__device__ __forceinline__ float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
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
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim] with interleave,
 *   return thread-local vector.
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
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

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_cache,
    float* __restrict__ sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
      cos.load(cos_cache + pos * rotary_dim + tx * vec_size);
      sin.load(sin_cache + pos * rotary_dim + tx * vec_size);
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

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_cache,
    float* __restrict__ sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
      cos.load(cos_cache + pos * rotary_dim + tx * vec_size);
      sin.load(sin_cache + pos * rotary_dim + tx * vec_size);
    }

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
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
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
    // apply rotary to k
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

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCache(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* cos_cache, float* sin_cache,
    IdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
    size_t k_stride_h, size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h, bool interleave, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&cos_cache,
                      (void*)&sin_cache,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h};
      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx,
                                                                DType, IdType>;

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 =
            BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size,
                                                                     bdx, DType, IdType>;
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
      }
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      auto kernel_0 =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;
      if (nblks_x >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsHeadParallelismKernel<INTERLEAVE, HEAD_DIM,
                                                                      vec_size, bdx, DType, IdType>;

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
      }
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotary(DType* q, DType* k, DType* q_rope, DType* k_rope,
                               IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n,
                               size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
                               size_t q_rope_stride_n, size_t q_rope_stride_h,
                               size_t k_rope_stride_n, size_t k_rope_stride_h, bool interleave,
                               float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
                                      size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
                                      size_t k_stride_h, bool interleave, float rope_scale,
                                      float rope_theta, cudaStream_t stream = nullptr) {
  return BatchQKApplyRotary<DType, IdType>(
      q, k, q, k, indptr, offsets, batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_stride_n, q_stride_h, k_stride_n,
      k_stride_h, interleave, rope_scale, rope_theta, stream);
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31Rotary(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
    size_t k_stride_h, size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h, bool interleave, float rope_scale, float rope_theta,
    float low_freq_factor, float high_freq_factor, float old_context_length,
    cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31RotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
    float high_freq_factor, float old_context_length, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks((nnz + bdy - 1) / bdy);
      dim3 nthrs(bdx, bdy);
      auto kernel =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
