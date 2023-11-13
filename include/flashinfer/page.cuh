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
#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

enum class AccessType {
  kProtective = 0U,    // Check whether page_iter is out of range
  kNonProtective = 1U  // Do not check whether page_iter is out of range
};

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, head_dim]
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  uint32_t num_layers;
  uint32_t layer_idx;
  uint32_t num_heads;
  uint32_t page_size;
  uint32_t head_dim;
  uint32_t batch_size;

  // [max_num_pages * num_layers * 2 * num_heads * page_size * head_dim]
  // The flattened key-value cache
  DType* data;
  // [batch_size + 1] The page indptr array, with the first element 0
  IdType* indptr;
  // [nnz_pages] The page indices array
  IdType* indices;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_offset;

  /* ------------ Auxliary Information Used in Cooperative Kernels ------------ */
  IdType* cooperative_indptr;
  IdType* batch_idx_map;
  IdType* chunk_start;
  IdType* seq_lens_before_split;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  __host__ __device__ __forceinline__ paged_kv_t()
      : num_layers(0),
        layer_idx(0),
        num_heads(0),
        page_size(0),
        head_dim(0),
        batch_size(0),
        data(nullptr),
        indptr(nullptr),
        indices(nullptr),
        last_page_offset(nullptr),
        cooperative_indptr(nullptr),
        batch_idx_map(nullptr),
        chunk_start(nullptr),
        seq_lens_before_split(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_layers The number of layers
   * \param layer_idx The index of the layer
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indptr The page indptr array
   * \param indices The page indices array
   * \param last_page_offset The offset of the last page for each request in the batch
   */
  __host__ __device__ __forceinline__ paged_kv_t(uint32_t num_layers, uint32_t layer_idx,
                                                 uint32_t num_heads, uint32_t page_size,
                                                 uint32_t head_dim, uint32_t batch_size,
                                                 DType* data, IdType* indptr, IdType* indices,
                                                 IdType* last_page_offset)
      : num_layers(num_layers),
        layer_idx(layer_idx),
        num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        data(data),
        indptr(indptr),
        indices(indices),
        last_page_offset(last_page_offset),
        cooperative_indptr(nullptr),
        batch_idx_map(nullptr),
        chunk_start(nullptr),
        seq_lens_before_split(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache with auxiliary information for cooperative kernels
   * \param num_layers The number of layers
   * \param layer_idx The index of the layer
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indptr The page indptr array
   * \param indices The page indices array
   * \param last_page_offset The offset of the last page for each request in the batch
   * \param cooperative_indptr The page indptr array for cooperative kernels
   * \param batch_idx_map The batch index map from for cooperative kernels
   * \param chunk_start The chunk start array for cooperative kernels
   * \param seq_lens_before_split The sequence length before split for cooperative kernels
   */
  __host__ __device__ __forceinline__ paged_kv_t(uint32_t num_layers, uint32_t layer_idx,
                                                 uint32_t num_heads, uint32_t page_size,
                                                 uint32_t head_dim, uint32_t batch_size,
                                                 DType* data, IdType* indptr, IdType* indices,
                                                 IdType* last_page_offset,
                                                 IdType* cooperative_indptr, IdType* batch_idx_map,
                                                 IdType* chunk_start, IdType* seq_lens_before_split)
      : num_layers(num_layers),
        layer_idx(layer_idx),
        num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        data(data),
        indptr(indptr),
        indices(indices),
        last_page_offset(last_page_offset),
        cooperative_indptr(cooperative_indptr),
        batch_idx_map(batch_idx_map),
        chunk_start(chunk_start),
        seq_lens_before_split(seq_lens_before_split) {}

  __host__ __device__ __forceinline__ size_t get_k_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx,
                                                               size_t feat_idx) const {
    return (((page_idx * num_layers + layer_idx) * 2 * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }

  __host__ __device__ __forceinline__ size_t get_v_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx,
                                                               size_t feat_idx) const {
    return ((((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }

  __host__ __device__ __forceinline__ uint32_t get_valid_page_size(uint32_t batch_idx,
                                                                   uint32_t page_iter) const {
    if (page_iter == indptr[batch_idx + 1] - 1) {
      return last_page_offset[batch_idx];
    } else {
      return page_size;
    }
  }

  template <AccessType access_type>
  __device__ __forceinline__ DType* get_k_ptr(uint32_t page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (access_type == AccessType::kProtective) {
      if (page_iter < indptr[batch_size]) {
        return data + get_k_elem_offset(indices[page_iter], head_idx, entry_idx, feat_idx);
      } else {
        return data;
      }
    } else {
      return data + get_k_elem_offset(indices[page_iter], head_idx, entry_idx, feat_idx);
    }
  }

  template <AccessType access_type>
  __device__ __forceinline__ DType* get_v_ptr(uint32_t page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (access_type == AccessType::kProtective) {
      if (page_iter < indptr[batch_size]) {
        return data + get_v_elem_offset(indices[page_iter], head_idx, entry_idx, feat_idx);
      } else {
        return data;
      }
    } else {
      return data + get_v_elem_offset(indices[page_iter], head_idx, entry_idx, feat_idx);
    }
  }
};

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam bdx The block dimension in x
 * \tparam bdy The block dimension in y
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, uint32_t bdy, typename DType,
          typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x / (num_heads / bdy);
  uint32_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_offset[batch_idx];

  uint32_t page_idx =
      paged_kv.indices[paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size];
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  vec_t<DType, vec_size>::memcpy(
      paged_kv.data + paged_kv.get_k_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size),
      key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  vec_t<DType, vec_size>::memcpy(
      paged_kv.data + paged_kv.get_v_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size),
      value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam bdx The block dimension in x
 * \tparam bdy The block dimension in y
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 */
template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, uint32_t bdy, typename DType,
          typename IdType>
__global__ void AppendPagedKVCachePrefillKernel(paged_kv_t<DType, IdType> paged_kv,
                                                DType* __restrict__ key, DType* __restrict__ value,
                                                IdType* __restrict__ append_indptr) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x / (num_heads / bdy);
  uint32_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_offset[batch_idx];
  uint32_t append_seq_len = append_indptr[batch_idx + 1] - append_indptr[batch_idx];
  uint32_t append_start = seq_len - append_seq_len;

#pragma unroll 2
  for (uint32_t j = 0; j < append_seq_len; ++j) {
    uint32_t page_seq_idx = j + append_start;
    uint32_t page_idx =
        paged_kv.indices[paged_kv.indptr[batch_idx] + page_seq_idx / paged_kv.page_size];
    uint32_t entry_idx = page_seq_idx % paged_kv.page_size;

    vec_t<DType, vec_size>::memcpy(
        paged_kv.data + paged_kv.get_k_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size),
        key + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);

    vec_t<DType, vec_size>::memcpy(
        paged_kv.data + paged_kv.get_v_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size),
        value + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);
  }
}

/*!
 * \brief CUDA kernel to convert the paged key-value cache to a ragged tensor
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam bdx The block dimension in x
 * \tparam bdy The block dimension in y
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param kv_indptr The indptr array of the ragged tensor
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, uint32_t bdy, typename DType,
          typename IdType>
__global__ void PagedKVCacheToRaggedTensorKernel(paged_kv_t<DType, IdType> paged_kv,
                                                 DType* __restrict__ key, DType* __restrict__ value,
                                                 IdType* __restrict__ kv_indptr) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x / (num_heads / bdy);
  uint32_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;

#pragma unroll 2
  for (uint32_t j = 0; j < kv_indptr[batch_idx + 1] - kv_indptr[batch_idx]; ++j) {
    uint32_t page_idx = paged_kv.indices[paged_kv.indptr[batch_idx] + j / paged_kv.page_size];
    uint32_t entry_idx = j % paged_kv.page_size;
    vec_t<DType, vec_size>::memcpy(
        key + ((kv_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_k_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size));
    vec_t<DType, vec_size>::memcpy(
        value + ((kv_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size,
        paged_kv.data + paged_kv.get_v_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size));
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                     cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the prefill phase
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCachePrefill(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                      IdType* append_indptr, cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCachePrefillKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&append_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Compute the index pointers of the ragged tensor converted from paged key-value
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param kv_indptr The indptr array of the ragged tensor (output)
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t PagedKVCacheToRaggedTensorComputeIndptr(paged_kv_t<DType, IdType> paged_kv,
                                                    std::vector<IdType>& kv_indptr_host,
                                                    cudaStream_t stream = nullptr) {
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t page_size = paged_kv.page_size;
  std::vector<IdType> paged_kv_indptr_host(batch_size + 1),
      paged_kv_last_page_offset_host(batch_size);
  kv_indptr_host.resize(batch_size + 1);

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(paged_kv_indptr_host.data(), paged_kv.indptr,
                                       sizeof(IdType) * (batch_size + 1), cudaMemcpyDeviceToHost,
                                       stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(paged_kv_last_page_offset_host.data(),
                                       paged_kv.last_page_offset, sizeof(IdType) * batch_size,
                                       cudaMemcpyDeviceToHost, stream));

  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));

  kv_indptr_host[0] = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    kv_indptr_host[i + 1] =
        kv_indptr_host[i] +
        (paged_kv_indptr_host[i + 1] - paged_kv_indptr_host[i] - 1) * page_size +
        paged_kv_last_page_offset_host[i];
  }

  return cudaSuccess;
}

/*!
 * \brief Convert the paged key-value cache to a ragged tensor
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param kv_indptr The indptr array of the ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t PagedKVCacheToRaggedTensor(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                       IdType* kv_indptr, cudaStream_t stream = nullptr) {
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t num_heads = paged_kv.num_heads;
  const uint32_t page_size = paged_kv.page_size;

  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = PagedKVCacheToRaggedTensorKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&kv_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
