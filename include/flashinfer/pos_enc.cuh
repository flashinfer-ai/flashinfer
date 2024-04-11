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
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset) {
  constexpr uint32_t head_dim = vec_size * bdx;
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);
  permuted_vec.cast_load(x + ((threadIdx.x * vec_size < head_dim / 2)
                                  ? threadIdx.x * vec_size + head_dim / 2
                                  : threadIdx.x * vec_size - head_dim / 2));

#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * freq[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos +
             ((threadIdx.x * vec_size < head_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
  return vec;
}

template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
__global__ void BatchQKApplyRotaryInPlaceKernel(DType* __restrict__ q, DType* __restrict__ k,
                                                IdType* __restrict__ indptr,
                                                IdType* __restrict__ offsets, uint32_t batch_size,
                                                uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    freq[i] =
        rope_rcp_scale *
        __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
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
        DType* q_ptr =
            q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                    indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0, seq_len, num_qo_heads);
        q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty);
        q_vec.cast_store(q_ptr + tx * vec_size);
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
        DType* k_ptr =
            k + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                    indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0, seq_len, num_kv_heads);
        k_vec = vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty);
        k_vec.cast_store(k_ptr + tx * vec_size);
      }
    }
  }
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t head_dim,
                                      float rope_scale = 1.f, float rope_theta = 1e4,
                                      cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = num_threads / bdx;
    dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
    dim3 nthrs(bdx, bdy);
    auto kernel = BatchQKApplyRotaryInPlaceKernel<HEAD_DIM, vec_size, bdx, DType, IdType>;
    void* args[] = {(void*)&q,
                    (void*)&k,
                    (void*)&indptr,
                    (void*)&offsets,
                    (void*)&batch_size,
                    (void*)&num_qo_heads,
                    (void*)&num_kv_heads,
                    (void*)&rope_rcp_scale,
                    (void*)&rope_rcp_theta};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
