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

#include <cooperative_groups.h>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../utils.cuh"
#include "state.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

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
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  float s_a_val = s_a[pos * num_heads + head_idx];
  float s_b_val = s_b[pos * num_heads + head_idx];
  float s_max = max(s_a_val, s_b_val);
  s_a_val = math::ptx_exp2(s_a_val - s_max);
  s_b_val = math::ptx_exp2(s_b_val - s_max);
  float a_scale = s_a_val / (s_a_val + s_b_val);
  float b_scale = s_b_val / (s_a_val + s_b_val);
  vec_t<float, vec_size> v_a_vec, v_b_vec, v_merged_vec;
  v_a_vec.cast_load(v_a + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_b_vec.cast_load(v_b + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_merged_vec[i] = a_scale * v_a_vec[i] + b_scale * v_b_vec[i];
  }
  v_merged_vec.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = math::ptx_log2(s_a_val + s_b_val) + s_max;
  }
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
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  float s_val = s[pos * num_heads + head_idx];
  float s_other_val = s_other[pos * num_heads + head_idx];
  float s_max = max(s_val, s_other_val);
  s_val = math::ptx_exp2(s_val - s_max);
  s_other_val = math::ptx_exp2(s_other_val - s_max);
  float scale = s_val / (s_val + s_other_val);
  float other_scale = s_other_val / (s_val + s_other_val);
  vec_t<float, vec_size> v_vec, v_other_vec;
  v_vec.cast_load(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_other_vec.cast_load(v_other + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec[i] = scale * v_vec[i] + other_scale * v_other_vec[i];
  }
  v_vec.cast_store(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s != nullptr) {
    s[pos * num_heads + head_idx] = math::ptx_log2(s_val + s_other_val) + s_max;
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                       float* s_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets.
 * \param vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param v The partial v of index sets. (n, num_index_sets, h, d)
 * \param s The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DTypeIn, typename DTypeOut>
__global__ void MergeStatesKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                  DTypeOut* __restrict__ v_merged, float* __restrict__ s_merged,
                                  uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;
  state_t<vec_size> st;

  vec_t<float, vec_size> v_merged_vec;
  v_merged_vec.fill(0.f);
#pragma unroll 2
  for (uint32_t iter = 0; iter < num_index_sets; ++iter) {
    float s = S[(pos * num_index_sets + iter) * num_heads + head_idx];
    vec_t<float, vec_size> v;
    v.cast_load(V + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim +
                tx * vec_size);
    st.merge(v, s, 1);
  }

  st.normalize();
  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets,
 *   accelerated for larget number of index sets.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param V The partial v of index sets. (n, num_index_sets, h, d)
 * \param S The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeOut>
__global__ void MergeStatesLargeNumIndexSetsKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                                   DTypeOut* __restrict__ v_merged,
                                                   float* __restrict__ s_merged,
                                                   uint32_t num_index_sets, uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = blockIdx.y;
  state_t<vec_size> st;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;

  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#pragma unroll
  for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
        V + ((pos * num_index_sets + (iter * bdy + ty)) * num_heads + head_idx) * head_dim +
            tx * vec_size,
        (iter * bdy + ty) < num_index_sets);
    cp_async::commit_group();
  }
#pragma unroll 4
  for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
    if (iter % bdx == 0) {
      s_smem[ty * bdx + tx] =
          iter * bdy + (ty * bdx + tx) < num_index_sets
              ? S[(pos * num_index_sets + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
              : 0.f;
      __syncthreads();
    }
    cp_async::wait_group<num_smem_stages - 1>();
    __syncthreads();
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
    if (iter * bdy + ty < num_index_sets) {
      float s = s_smem[(iter % bdx) * bdy + ty];
      st.merge(v, s, 1);
    }
    __syncthreads();
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
        V +
            ((pos * num_index_sets + ((iter + num_smem_stages) * bdy + ty)) * num_heads +
             head_idx) *
                head_dim +
            tx * vec_size,
        (iter + num_smem_stages) * bdy + ty < num_index_sets);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  __syncthreads();

  st.normalize();
  threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
  st.normalize();

  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel to merge self-attention states of multiple index sets, the number of index
 *   sets at each position might vary.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeOut The data type of v_merged.
 * \param V The partial v of index sets. (nnz, h, d)
 * \param S The logsumexp value of index sets. (nnz, h)
 * \param indptr The start offsets of each position in the variable length array.
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeOut, typename IdType>
__global__ void VariableLengthMergeStatesKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                                IdType* indptr, DTypeOut* __restrict__ v_merged,
                                                float* __restrict__ s_merged, uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = blockIdx.y;
  state_t<vec_size> st;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;

  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));
  const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

#pragma unroll
  for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
        V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
        (iter * bdy + ty) < num_index_sets);
    cp_async::commit_group();
  }
#pragma unroll 4
  for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
    if (iter % bdx == 0) {
      s_smem[ty * bdx + tx] =
          iter * bdy + (ty * bdx + tx) < num_index_sets
              ? S[(indptr[pos] + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
              : 0.f;
      __syncthreads();
    }
    cp_async::wait_group<num_smem_stages - 1>();
    __syncthreads();
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
    if (iter * bdy + ty < num_index_sets) {
      float s = s_smem[(iter % bdx) * bdy + ty];
      st.merge(v, s, 1);
    }
    __syncthreads();
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
        V +
            ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                head_dim +
            tx * vec_size,
        (iter + num_smem_stages) * bdy + ty < num_index_sets);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  __syncthreads();

  st.normalize();
  threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
  st.normalize();

  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
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
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeOut* v_merged,
                       float* s_merged, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
                       cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
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
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <typename DType>
cudaError_t MergeStateInPlace(DType* v, float* s, DType* v_other, float* s_other, uint32_t seq_len,
                              uint32_t num_heads, uint32_t head_dim,
                              cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
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
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note s are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t MergeStates(DTypeIn* v, float* s, DTypeOut* v_merged, float* s_merged,
                        uint32_t num_index_sets, uint32_t seq_len, uint32_t num_heads,
                        uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    if (num_index_sets >= seq_len) {
      constexpr uint32_t num_threads = 128;
      constexpr uint32_t bdy = num_threads / bdx;
      dim3 nblks(seq_len, num_heads);
      dim3 nthrs(bdx, bdy);
      constexpr uint32_t num_smem_stages = 4;
      auto kernel = MergeStatesLargeNumIndexSetsKernel<vec_size, bdx, bdy, num_smem_stages, DTypeIn,
                                                       DTypeOut>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads};
      uint32_t smem_size =
          num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      uint32_t bdy = num_heads;
      dim3 nblks(seq_len);
      dim3 nthrs(bdx, bdy);
      auto kernel = MergeStatesKernel<vec_size, DTypeIn, DTypeOut>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads, &head_dim};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    }
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeOut* v_merged,
                                      float* s_merged, uint32_t seq_len, uint32_t num_heads,
                                      uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    dim3 nblks(seq_len, num_heads);
    dim3 nthrs(bdx, bdy);
    constexpr uint32_t num_smem_stages = 4;
    auto kernel = VariableLengthMergeStatesKernel<vec_size, bdx, bdy, num_smem_stages, DTypeIn,
                                                  DTypeOut, IdType>;
    void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &num_heads};
    uint32_t smem_size =
        num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_CASCADE_CUH_
