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
 * \tparam bdx The block size of x dimension.
 * \tparam bdy The block size of y dimension.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeOut The data type of v_merged.
 * \param s_a The partial v of index set A. (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn, typename DTypeOut>
__global__ void MergeStateKernel(DTypeIn* __restrict__ v_a, float* __restrict__ s_a,
                                 DTypeIn* __restrict__ v_b, float* __restrict__ s_b,
                                 DTypeOut* __restrict__ v_merged, uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t batch_idx = blockIdx.x / (num_heads / bdy);
  uint32_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;
  constexpr uint32_t head_dim = bdx * vec_size;

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
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets.
 * \param bdx The block size of x dimension.
 * \param bdy The block size of y dimension.
 * \param vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param v The partial v of index sets. (num_index_sets, n, h, d)
 * \param s The logsumexp value of index sets. (num_index_sets, n, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \note s are logsumexp values with base 2.
 */
template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn, typename DTypeOut>
__global__ void MergeStatesKernel(DTypeIn* __restrict__ v, float* __restrict__ s,
                                  DTypeOut* __restrict__ v_merged, uint32_t num_index_sets,
                                  uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t batch_idx = blockIdx.x / (num_heads / bdy);
  uint32_t batch_size = gridDim.x / (num_heads / bdy);
  uint32_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;
  constexpr uint32_t head_dim = bdx * vec_size;
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
 * \param batch_size The batch size of self-attention states.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeOut* v_merged,
                       uint32_t batch_size, uint32_t num_heads, uint32_t head_dim,
                       cudaStream_t stream = nullptr) {
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateKernel<bdx, bdy, vec_size, DTypeIn, DTypeOut>;
    void* args[] = {&v_a, &s_a, &v_b, &s_b, &v_merged, &num_heads};
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
 * \param num_index_sets The number of index sets.
 * \param batch_size The batch size of self-attention states.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeStates(DTypeIn* v, float* s, DTypeOut* v_merged, uint32_t num_index_sets,
                        uint32_t batch_size, uint32_t num_heads, uint32_t head_dim,
                        cudaStream_t stream = nullptr) {
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStatesKernel<bdx, bdy, vec_size, DTypeIn, DTypeOut>;
    void* args[] = {&v, &s, &v_merged, &num_index_sets, &num_heads};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_CASCADE_CUH_
