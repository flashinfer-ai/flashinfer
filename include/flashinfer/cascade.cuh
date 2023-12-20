/*!
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
#ifndef FLASHINFER_CASCADE_CUH_
#define FLASHINFER_CASCADE_CUH_

#include "decode.cuh"
#include "math.cuh"
#include "prefill.cuh"
#include "state.cuh"
#include "utils.cuh"

namespace flashinfer {

/*!
 * \brief The CUDA kernel that merges the self-attention state of two index sets A and B.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeOut The data type of v_merged.
 * \param v_a The partial v of index set A. (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DTypeIn, typename DTypeOut>
__global__ void MergeStateKernel(DTypeIn* __restrict__ v_a, float* __restrict__ s_a,
                                 DTypeIn* __restrict__ v_b, float* __restrict__ s_b,
                                 DTypeOut* __restrict__ v_merged, float* __restrict__ s_merged,
                                 uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  float s_a_val = s_a[batch_idx * num_heads + head_idx];
  float s_b_val = s_b[batch_idx * num_heads + head_idx];
  float s_max = max(s_a_val, s_b_val);
  s_a_val = math::ptx_exp2(s_a_val - s_max);
  s_b_val = math::ptx_exp2(s_b_val - s_max);
  float a_scale = s_a_val / (s_a_val + s_b_val);
  float b_scale = s_b_val / (s_a_val + s_b_val);
  vec_t<float, vec_size> v_vec_a, v_vec_b, v_vec_o;
  v_vec_a.cast_load(v_a + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  v_vec_b.cast_load(v_b + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec_o[i] = a_scale * v_vec_a[i] + b_scale * v_vec_b[i];
  }
  v_vec_o.cast_store(v_merged + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  s_merged[batch_idx * num_heads + head_idx] = math::ptx_log2(s_a_val + s_b_val) + s_max;
}

/*!
 * \brief The CUDA kernel that merges the self-attention state with another state in-place.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DType>
__global__ void MergeStateInPlaceKernel(DType* __restrict__ v, float* __restrict__ s,
                                        DType* __restrict__ v_other, float* __restrict__ s_other,
                                        uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  float s_val = s[batch_idx * num_heads + head_idx];
  float s_other_val = s_other[batch_idx * num_heads + head_idx];
  float s_max = max(s_val, s_other_val);
  s_val = math::ptx_exp2(s_val - s_max);
  s_other_val = math::ptx_exp2(s_other_val - s_max);
  float scale = s_val / (s_val + s_other_val);
  float other_scale = s_other_val / (s_val + s_other_val);
  vec_t<float, vec_size> v_vec, v_other_vec;
  v_vec.cast_load(v + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  v_other_vec.cast_load(v_other + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec[i] = scale * v_vec[i] + other_scale * v_other_vec[i];
  }
  v_vec.cast_store(v + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  s[batch_idx * num_heads + head_idx] = math::ptx_log2(s_val + s_other_val) + s_max;
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets.
 * \param vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param v The partial v of index sets. (num_index_sets, n, h, d)
 * \param s The logsumexp value of index sets. (num_index_sets, n, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DTypeIn, typename DTypeOut>
__global__ void MergeStatesKernel(DTypeIn* __restrict__ v, float* __restrict__ s,
                                  DTypeOut* __restrict__ v_merged, float* __restrict__ s_merged,
                                  uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t batch_size = gridDim.x;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;
  float m = -5e4, d = 0.f;

  vec_t<float, vec_size> v_vec_o;
  v_vec_o.fill(0.f);
  for (uint32_t j = 0; j < num_index_sets; ++j) {
    float s_val = s[(j * batch_size + batch_idx) * num_heads + head_idx];
    vec_t<float, vec_size> v_j;
    v_j.cast_load(v + ((j * batch_size + batch_idx) * num_heads + head_idx) * head_dim +
                  tx * vec_size);
    float m_prev = m;
    m = max(m, s_val);
    float o_scale = math::ptx_exp2(m_prev - m);
    float v_j_scale = math::ptx_exp2(s_val - m);
    d = d * o_scale + v_j_scale;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v_vec_o[i] = v_vec_o[i] * o_scale + v_j[i] * v_j_scale;
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec_o[i] = __fdividef(v_vec_o[i], d);
  }
  v_vec_o.cast_store(v_merged + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
  s_merged[batch_idx * num_heads + head_idx] = math::ptx_log2(d) + m;
}

/*!
 * \brief Merge the self-attention state of two index sets A and B.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeOut The data type of v_merged.
 * \param v_a The partial v of index set A (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param batch_size The batch size of self-attention states.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeOut* v_merged,
                       float* s_merged, uint32_t batch_size, uint32_t num_heads, uint32_t head_dim,
                       cudaStream_t stream = nullptr) {
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateKernel<vec_size, DTypeIn, DTypeOut>;
    void* args[] = {&v_a, &s_a, &v_b, &s_b, &v_merged, &s_merged, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge the self-attention state with another state in place.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param batch_size The batch size of self-attention states.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <typename DType>
cudaError_t MergeStateInPlace(DType* v, float* s, DType* v_other, float* s_other,
                              uint32_t batch_size, uint32_t num_heads, uint32_t head_dim,
                              cudaStream_t stream = nullptr) {
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateInPlaceKernel<vec_size, DType>;
    void* args[] = {&v, &s, &v_other, &s_other, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge self-attention states of a list of index sets.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param v The partial v of index sets. (num_index_sets, n, h, d)
 * \param s The logsumexp value of index sets. (num_index_sets, n, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_index_sets The number of index sets.
 * \param batch_size The batch size of self-attention states.
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note s are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeStates(DTypeIn* v, float* s, DTypeOut* v_merged, float* s_merged,
                        uint32_t num_index_sets, uint32_t batch_size, uint32_t num_heads,
                        uint32_t head_dim, cudaStream_t stream = nullptr) {
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStatesKernel<vec_size, DTypeIn, DTypeOut>;
    void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_CASCADE_CUH_
